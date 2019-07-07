#include <stdint.h>
#include <stdio.h>              /* perror */
#include <stdlib.h>             /* posix_memalign */
#include <math.h>               /* fmod, pow */
#include <time.h>               /* time() */
#include <string.h>             /* memcpy() */
#include <limits.h>
#include <sys/time.h>           /* gettimeofday */
#include <errno.h>
#include <cuda_runtime.h> 

#include "rdtsc.h"
#include "dataload.h"

///////////////////// Macro Define //////////////////////
#define random(x) (rand()%x)
#define nstream 4

#define THREAD_NUM 1024 //4096
#define BLOCK_NUM 13
#define DEVICE_MAX_COUNT 8

#define num_part 15000
#define num_supplier 2000
#define num_customer 30000
#define num_date 2555
////////////////// End of Macro Define ///////////////////

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
////////// End of Global Variables /////////////////////

/////////////// Function Declarations /////////////////////
int64_t STARJOIN_CU(column_t *factT, vector_t *DimVec, vector_para *VecParams,
    vector_t *MIndex, int nstreams, int filterflag);
int create_vectors_pk(vector_t * DimVec, vector_para *VecParams, int group_num = 100);
int load_fact_fk(column_t * FactColumns,vector_para *VecParams, int *lineorder_size);
void print_timing(uint64_t total, uint64_t numtuples);
/////////////// End of Function Declarations ///////////////////

//////////////////// Tool Functions /////////////////////
// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline cudaError_t checkCuda(cudaError_t result) {
#if defined(DEBUG) || defined(_DEBUG)
	if (result != cudaSuccess) {
		fprintf(stderr, "CUDA Runtime Error: %s\n",
			cudaGetErrorString(result));
		assert(result == cudaSuccess);
	}
#endif
	return result;
}

///////////////// End of Tool Functions ///////////////////

typedef struct {
  intkey_t *dev_data;
  vectorkey_t *dev_DimVec;
  vectorkey_t *dev_MIndex;
  int *results;
  int *dev_results[nstream];
  int vec_len, data_len;
  cudaStream_t streams[nstream];
  uint64_t timer;
} device_args_t;


int main(int argc, char **argv)
{
  int64_t results; 
  int nstreams = 40;
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
    MeasureIndex.column[i]=0;

  for(int j = 0; j < 4; j++) {
    firstfilterflag = j;
    if(VecParas[j].selectivity!=0)
	    results = STARJOIN_CU(&FactColumns[j], &DimVector[j], &VecParas[j], &MeasureIndex, nstreams, firstfilterflag);
  }

  return 0;
}

static __global__ void VecFK_FilterJoin(intkey_t *a, vectorkey_t *b,  vectorkey_t *c, int num_lineorder, int fkid) {
	const int tid = threadIdx.x;
	const int bid = blockIdx.x;
  
  if (fkid == 0) {
    for (int i = bid * THREAD_NUM + tid; i < num_lineorder; i += BLOCK_NUM * THREAD_NUM) {
      c[i] = a[b[i]];
    }
  } else {
    for (int i = bid * THREAD_NUM + tid; i < num_lineorder; i += BLOCK_NUM * THREAD_NUM) {
      if(c[i] != -1) c[i] = a[b[i]];
    }
  }
}

int64_t STARJOIN_CU(column_t *factT, vector_t *DimVec, vector_para *VecParams, vector_t *MIndex, int nstreams, int filterflag) {
  intkey_t *data = factT->column;
  int n = factT->num_tuples;
  int fk_id = filterflag;
  int vec_len = VecParams[fk_id].num_tuples;
  vectorkey_t *dimVec = DimVec->column;
  vectorkey_t *mIdx = MIndex->column;
  int64_t result = 0;

  uint64_t timer1;
  
  device_args_t deviceArgs[DEVICE_MAX_COUNT];
  int cudaDeviceNum = 0;
  // get number of device
  checkCuda(cudaGetDeviceCount(&cudaDeviceNum));

  int dataBlock = n / cudaDeviceNum; 
  int dataRemSize = n;
  intkey_t *tmpPtr = data;
  vectorkey_t *mdx_cp = mIdx;

  for (int i = 0; i < cudaDeviceNum; i++) {
    int dataSize = (i == (cudaDeviceNum - 1) ? dataRemSize : dataBlock);
    checkCuda(cudaSetDevice(i));

    dataRemSize -= dataSize;

    checkCuda(cudaMalloc((void**)&(deviceArgs[i].dev_data), sizeof(intkey_t) * dataSize));
    checkCuda(cudaMalloc((void**)&(deviceArgs[i].dev_MIndex), sizeof(vectorkey_t) * dataSize));
    checkCuda(cudaMalloc((void**)&(deviceArgs[i].dev_DimVec), sizeof(vectorkey_t) * vec_len));
  } 

  dataRemSize = n;
  for (int i = 0; i < cudaDeviceNum; i++) {
    int dataSize = (i == (cudaDeviceNum - 1) ? dataRemSize : dataBlock);
    checkCuda(cudaSetDevice(i));

    dataRemSize -= dataSize;
    checkCuda(cudaMemcpy(deviceArgs[i].dev_data, tmpPtr, sizeof(intkey_t)
          * dataSize, cudaMemcpyHostToDevice)); 

    checkCuda(cudaMemcpy(deviceArgs[i].dev_MIndex, mdx_cp,
          sizeof(vectorkey_t) * dataSize, cudaMemcpyHostToDevice)); 

    tmpPtr += dataSize;
    mdx_cp += dataSize;

    // Time for DimVec transfering and GPU computing
    checkCuda(cudaMemcpy(deviceArgs[i].dev_DimVec, dimVec,
          sizeof(vectorkey_t) * vec_len,  cudaMemcpyHostToDevice)); 
  }

  dataRemSize = n;

  startTimer(&timer1);
  for (int i = 0; i < cudaDeviceNum; i++) {
    int dataSize = (i == (cudaDeviceNum - 1) ? dataRemSize : dataBlock);
    checkCuda(cudaSetDevice(i));

    int numDatastr = dataSize / nstream;
    int numData = dataSize;
    dataRemSize -= dataSize;
    
    int tmpDataLen = 0;
    for (int j = 0; j < nstream; j++) {
      tmpDataLen = (j == (nstream - 1) ? numData : numDatastr);
      numData -= numDatastr;
      checkCuda(cudaStreamCreate(&deviceArgs[i].streams[j]));
      VecFK_FilterJoin<<<BLOCK_NUM, THREAD_NUM, 2 * THREAD_NUM * sizeof(float),
        deviceArgs[i].streams[j]>>>(deviceArgs[i].dev_data + j * numDatastr,
            deviceArgs[i].dev_DimVec, deviceArgs[i].dev_MIndex + j * numDatastr, 
            tmpDataLen, fk_id);
      cudaError_t cudaStatus = cudaGetLastError();
      if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
      }
    }
  }

  mdx_cp = mIdx;
  dataRemSize = n;
  // sync all device
  for (int i = 0; i < cudaDeviceNum; i++) {
    checkCuda(cudaSetDevice(i));
    for (int j = 0; j < nstream; j++) {
      checkCuda(cudaStreamSynchronize(deviceArgs[i].streams[j]));
    }
    int dataSize = (i == (cudaDeviceNum - 1) ? dataRemSize : dataBlock);
    checkCuda(cudaMemcpy(mdx_cp, deviceArgs[i].dev_MIndex,
          sizeof(vectorkey_t) * dataSize, cudaMemcpyDeviceToHost)); 
    mdx_cp += dataSize;
    dataRemSize -= dataSize;
  }

  // free all device memory
  for (int i = 0; i < cudaDeviceNum; i++) {
    checkCuda(cudaSetDevice(i));
    for (int j = 0; j < nstream; j++) {
      checkCuda(cudaStreamDestroy(deviceArgs[i].streams[j]));
    }
    checkCuda(cudaFree(deviceArgs[i].dev_data));
    checkCuda(cudaFree(deviceArgs[i].dev_MIndex));
    checkCuda(cudaFree(deviceArgs[i].dev_DimVec));
  }
  
  stopTimer(&timer1); 
  print_timing(timer1, n);

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

void print_timing(uint64_t total, uint64_t numtuples) {
    double cyclestuple = total;
    cyclestuple /= numtuples;
    fprintf(stdout, "TOTAL-TUPLES  ,RESULT ,RUNTIME TOTAL ,TOTAL-TIME-USECS,  CYCLES-PER-TUPLE: \n");
    fprintf(stderr, "%-15llu%11.4lf    %11.4lf", numtuples, total, cyclestuple);
    fflush(stdout);
    fflush(stderr);
    fprintf(stdout, "\n");
}
