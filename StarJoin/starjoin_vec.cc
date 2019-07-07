#include <stdio.h>              /* perror */
#include <stdlib.h>             /* posix_memalign */
#include <math.h>               /* fmod, pow */
#include <time.h>               /* time() */
#include <string.h>             /* memcpy() */
#include <limits.h>
#include <pthread.h>
#include <sys/time.h>           /* gettimeofday */
#include "rdtsc.h"
#include "dataload.h"

///////////////////// Macro Define //////////////////////
#define num_part 1500000
#define num_supplier 200000
#define num_customer 3000000
#define num_date 2555
#define default_vec_len 200000

#ifndef BARRIER_ARRIVE
/** barrier wait macro */
#define BARRIER_ARRIVE(B, RV)                            \
    RV = pthread_barrier_wait(B);                       \
    if(RV !=0 && RV != PTHREAD_BARRIER_SERIAL_THREAD){  \
        printf("Couldn't wait on barrier\n");           \
        exit(EXIT_FAILURE);                             \
    }
#endif

////////////////// End of Macro Define ///////////////////


//////////////////// Custom Types and functions //////////////////// 
typedef struct arg_sj arg_sj; 
//starjoin struct by zys
struct arg_sj {
    int32_t             tid;
    int32_t             fkid;
    column_t *          fks;
    vector_t *          pks;
    vectorkey_t *          MInx;
    int32_t             FKStartIndex;
    int32_t             MIStartIndex;
    pthread_barrier_t * barrier;
    int64_t             num_results;
#ifndef NO_TIMING
    /* stats about the thread */
    uint64_t timer1;
    struct timeval start, end;
#endif
};

//////////////////// End of Custom Types //////////////////// 

//////////////////// Global Variables //////////////////// 
column_t FactColumns[4];
vector_t DimVector[4];
vector_t MeasureIndex;
double time_sum = 0.0;

static struct vector_para VecParas [] = 
{
  {1, 5, num_date, 1},  
  {1, 5, num_supplier, 1},   //--0 represents bitmap filtering, non-0 represents vector filter by zys
  {1, 5, num_part, 1},
  {1, 5, num_customer, 1},
  {0, 0, 0, 0}
};

int startOffsets[4];
/////////////// End of Global Variables /////////////////////

/////////////// Function Declarations /////////////////////
int64_t STARJOIN(column_t *factT, vector_t *DimVec, vector_t *MIndex,int nthreads, vector_para *parame, int *filterflag, int vec_len = default_vec_len);
int create_vectors_pk(vector_t * DimVec, vector_para *VecParams, int group_num = 100);
int load_fact_fk(column_t * FactColumns,vector_para *VecParams, int *lineorder_size);
void print_timing(uint64_t total, uint64_t numtuples, int64_t result, struct timeval * start, struct timeval * end);
/////////////// End of Function Declarations ///////////////////

int main(int argc, char **argv)
{
  int64_t results = 0; 
  int nthreads = 40;
  int num_lineorder;
  int firstfilterflag = 0;
  int group_num = 128;

  load_fact_fk(FactColumns, VecParas, &num_lineorder);

  cout << num_lineorder << " tuples of Fact table loaded!" << endl;

  MeasureIndex.column = (vectorkey_t*) malloc(num_lineorder * sizeof(vectorkey_t));

  if (!MeasureIndex.column) {
    perror("out of memory when creating fact Measure Index vector.");
    return -1;
  }

  create_vectors_pk(DimVector, VecParas, group_num);
  for(int i = 0; i < num_lineorder; i++) 
    MeasureIndex.column[i] = 0;

  for(int j = 0; j < 4; j++) {
    firstfilterflag = j;
    if(VecParas[j].selectivity != 0)
	    results += STARJOIN(&FactColumns[j], &DimVector[j], &MeasureIndex, nthreads, &VecParas[j], &firstfilterflag);
  }

  printf("Time sum: %11.4lf\n", time_sum);
  printf("Result sum: %d\n", results);

  return 0;
}
				
void *STARJOIN_thread(void * param) {
  int rv;
  int fk_id;
  arg_sj * args = (arg_sj*) param;
  fk_id = args->fkid; //printf("\nfk_id:%d\n",fk_id);

  /* wait at a barrier until each thread starts and start timer */
  BARRIER_ARRIVE(args->barrier, rv);

#ifndef NO_TIMING
  /* the first thread checkpoints the start time */
  if(args->tid == 0){
      gettimeofday(&args->start, NULL);
      startTimer(&args->timer1);
  }
#endif
  BARRIER_ARRIVE(args->barrier, rv);

  int64_t matches = 0;
  int idx;
  if (fk_id == 0) {
    for (uint32_t i = 0; i < args->fks->num_tuples; i++) {
      int fkIndex = args->fks->column[args->FKStartIndex+i] - startOffsets[fk_id];
      idx = args->pks->column[fkIndex];
      args->MInx[args->MIStartIndex + i] = idx;
      if (idx != -1) matches++;
    }
  } else {
    for (uint32_t i = 0; i < args->fks->num_tuples; i++) {
      if (args->MInx[args->MIStartIndex + i] != -1) {
        int fkIndex = args->fks->column[args->FKStartIndex+i] - startOffsets[fk_id];
        idx = args->pks->column[fkIndex];
        args->MInx[args->MIStartIndex + i] = idx;
        if (idx != -1) matches++;
      }
    }
  }
  args->num_results = matches;

  /* for a reliable timing we have to wait until all finishes */
  BARRIER_ARRIVE(args->barrier, rv);

  /* probe phase finished, thread-0 checkpoints the time */
  if(args->tid == 0){
    stopTimer(&args->timer1); 
    gettimeofday(&args->end, NULL);
  }

  return 0;
}

int64_t STARJOIN(column_t *factT, vector_t *DimVec, vector_t *MIndex,int nthreads, vector_para *parame, int *filterflag, int vec_len)
{
    int64_t result = 0;
    int32_t numS, numSthr, FactTuples; //numR,numRthr --total and per thread num 
    int rv;
  	FactTuples = factT->num_tuples;
    int vec_num = (FactTuples + vec_len - 1) / vec_len;
    int tmpFactTuples = FactTuples;
    arg_sj args[nthreads];


    for (int j = 0; j < vec_num; j++) {
      int fact_len = (j == vec_num - 1) ? tmpFactTuples : vec_len;
      tmpFactTuples -= vec_len;
      numS = fact_len;
      numSthr = numS / nthreads;
      pthread_t tid[nthreads];
      pthread_attr_t attr;
      pthread_barrier_t barrier;
      rv = pthread_barrier_init(&barrier, NULL, nthreads);
      if(rv != 0){
          printf("Couldn't create the barrier\n");
          exit(EXIT_FAILURE);
      }
      pthread_attr_init(&attr);
      for (int i = 0; i < nthreads; i++) {
          args[i].tid = i;
          args[i].fkid = *filterflag;
          args[i].fks = factT;
          args[i].pks = DimVec;
          args[i].MInx = MIndex->column;
          args[i].FKStartIndex = 0;
          args[i].MIStartIndex = 0;
          args[i].barrier = &barrier;

          args[i].fks->num_tuples = (i == (nthreads - 1)) ? numS : numSthr;
          args[i].FKStartIndex += numSthr * i;        
          args[i].MIStartIndex += numSthr * i;
          numS -= numSthr;

          rv = pthread_create(&tid[i], &attr, STARJOIN_thread, (void*)&args[i]);
          if (rv) {
              printf("ERROR; return code from pthread_create() is %d\n", rv);
              exit(-1);
          }
      }
      for(int i = 0; i < nthreads; i++) {
          pthread_join(tid[i], NULL);
          //-- sum up results
          result += args[i].num_results;
      }
    }

#ifndef NO_TIMING
    //-- now print the timing results: 
    print_timing(args[0].timer1, FactTuples, result, &args[0].start, &args[0].end);
#endif

    return result;
}

int create_vectors_pk(vector_t * DimVec, vector_para *VecParams, int group_num)
{
  time_t t;
  srand((unsigned) time(&t));

  for (int i = 0; i < 4; i++) {
    DimVec[i].column= (vectorkey_t*) malloc(VecParams[i].num_tuples * sizeof(vectorkey_t));
    int NaN_num = VecParams[i].selectivity * VecParams[i].num_tuples;
    
    int offset = VecParams[i].num_tuples / NaN_num;
    int nan_id = 0;
    for (int j = 0; j < VecParams[i].num_tuples; j++) {
      DimVec[i].column[j] = -1;  
      if (nan_id < VecParams[i].num_tuples)
        DimVec[i].column[nan_id] = rand() % group_num;  
      nan_id += offset;
    }
  }

  return 0;
}

int load_fact_fk(column_t * FactColumns,vector_para *VecParams, int *lineorder_size) {
    printf("\nloading fact fk columns...\n");
    char file_number;
    char *filename = (char *) malloc(sizeof(char) * 42);

    FILE *dataset;

    strcpy(filename, filepath);

    cout << "Choose a data file (1. lineorder_1.tbl; 2. lineorder_10.tbl): ";
    cin >> file_number;
    if (file_number == '2') {
        strcat(filename, name2);
    }
    else {
        strcat(filename, name1);
    }

    dataset = fopen(filename, "r");
    if (dataset == NULL) {
        cout << "[error] cannot load table\n";
        exit(1);
    }

    table_info *info = (table_info *) malloc(sizeof(table_info));
    info->rows = 0;
    info->table = (lineorder_struct *) malloc(sizeof(lineorder_struct));
    info->stat.lo_custkey_min = INT_MAX;
    info->stat.lo_custkey_max = 0;
    info->stat.lo_partkey_min = INT_MAX;
    info->stat.lo_partkey_max = 0;
    info->stat.lo_suppkey_min = INT_MAX;
    info->stat.lo_suppkey_max = 0;
    info->stat.lo_startdate = INT_MAX;

    loadTable(dataset, info);
    startOffsets[0] = info->stat.lo_startdate;
    startOffsets[1] = info->stat.lo_suppkey_min;
    startOffsets[2] = info->stat.lo_partkey_min;
    startOffsets[3] = info->stat.lo_custkey_min;

    VecParams[0].num_tuples = info->stat.lo_enddate - info->stat.lo_startdate; 
    VecParams[1].num_tuples = info->stat.lo_suppkey_max - info->stat.lo_suppkey_min; 
    VecParams[2].num_tuples = info->stat.lo_partkey_max - info->stat.lo_partkey_min; 
    VecParams[3].num_tuples = info->stat.lo_custkey_max - info->stat.lo_custkey_min; 

    cout << "[info] table loaded\n";

    column_t tmpcol;
    intkey_t * tmp;
    int factrows = info->rows;
    *lineorder_size = factrows;

    for (int i = 0; i < 4; i++) {
      if(VecParams[i].selectivity != 0) {
        FactColumns[i].column= (intkey_t*) malloc(factrows * sizeof(intkey_t));
        if (!FactColumns[i].column) { 
          perror("out of memory when creating fact fk column.");
          return -1; 
        }
        FactColumns[i].num_tuples = factrows;
      if (i == 0) {
        memcpy(FactColumns[i].column, info->table->lo_normalorderdate, sizeof(intkey_t) * factrows);
      } else if (i == 1) {
        memcpy(FactColumns[i].column, info->table->lo_suppkey, sizeof(intkey_t) * factrows);
      } else if (i == 2) {
        memcpy(FactColumns[i].column, info->table->lo_partkey, sizeof(intkey_t) * factrows);
      } else if (i == 3) {
        memcpy(FactColumns[i].column, info->table->lo_custkey, sizeof(intkey_t) * factrows);
      }

     }//--end for if(VecParams[i].selectivity>0)    by zys 
    }//--end for table loop by zys

    fclose(dataset);
    return 0;
}

void print_timing(uint64_t total, uint64_t numtuples, int64_t result, struct timeval * start, struct timeval * end)
{
    double diff_usec = (((*end).tv_sec*1000000L + (*end).tv_usec)
                        - ((*start).tv_sec*1000000L+(*start).tv_usec));
    double cyclestuple = total;
    cyclestuple /= numtuples;
    fprintf(stdout, "TOTAL-TUPLES  ,RESULT ,RUNTIME TOTAL ,TOTAL-TIME-USECS,  CYCLES-PER-TUPLE: \n");
    fprintf(stderr, "%-15llu%-15llu%-15llu%11.4lf    %11.4lf", numtuples, result, total,  diff_usec, cyclestuple);
    time_sum += diff_usec;
    fflush(stdout);
    fflush(stderr);
    fprintf(stdout, "\n");
}
