/* $begin mountainmain */
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>           /* gettimeofday */
#include <pthread.h>
#include <errno.h>
#include <cuda_runtime.h> 
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if !defined(__i386__) && !defined(__x86_64__) && !defined(__sparc__)
#warning No supported architecture found -- timers will return junk.
#endif

static __inline__ uint64_t curtick() {
	uint64_t tick;
#if defined(__i386__)
	unsigned long lo, hi;
	__asm__ __volatile__ (".byte 0x0f, 0x31" : "=a" (lo), "=d" (hi));
	tick = (uint64_t) hi << 32 | lo;
#elif defined(__x86_64__)
	unsigned long lo, hi;
	__asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
	tick = (uint64_t) hi << 32 | lo;
#elif defined(__sparc__)
	__asm__ __volatile__ ("rd %%tick, %0" : "=r" (tick));
#endif
	return tick;
}

static __inline__ void startTimer(uint64_t* t) {
	*t = curtick();
}

static __inline__ void stopTimer(uint64_t* t) {
	*t = curtick() - *t;
}

#ifdef __cplusplus
}
#endif

#define L1 (1<<15)    /* Working set size for L1 cache 32KB */
#define L2 (1<<18)    /* Working set size for L2 cache 256KB */
#define L3 (1<<20)*2.5    /* Working set size for L3 cache 2.5MB */
#define LLC (1<<20)*55    /* Working set size for LLC cache 55MB */
#define MAXELEMS 600
#define random(x) (rand()%x)
#define nthread 4
#define nstream 4

#define THREAD_NUM 1024 //4096
#define BLOCK_NUM 13
#define DEVICE_MAX_COUNT 8

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
	if (result != cudaSuccess) {
		fprintf(stderr, "CUDA Runtime Error: %s\n",
			cudaGetErrorString(result));
		assert(result == cudaSuccess);
	}
#endif
	return result;
}

typedef struct {
  int *dev_data, *dev_vector, *results;
  int *dev_results[nstream];
  int vec_len, data_len;
  cudaStream_t streams[nstream];
  uint64_t timer;
} device_args_t;

void init_data(int *data, int n, int cardinality);
double run_on_gpu(int *data, int n, int *vector, int vec_len, bool hasTransferTime);

void generator(sample_t *pk_table, int fk_len, sample_t *fk_table, int pk_len) {
  int i, j;
  for (i = 0; i < fk_len; i++) {
    fk_table[i].rowkey = i;
    fk_table[i].device_id = random(9);
    for (j = 0; j < 128; j++)
      fk_table[i].model[j] = 0;
  }
  for (i = 0; i < pk_len; i++){
    pk_table[i].rowkey = random(fk_len);
    pk_table[i].device_id = random(9);
    for (j = 0; j < 128; j++)
      pk_table[i].model[j] = 0;
  }
}


/* $begin mountainmain */
int main()
{
    int i, pk_len =L1/4*20/100, fk_len = MAXELEMS;

    sample_t* fk_table =(sample_t*)malloc(sizeof(sample_t) * fk_len);      /* foreign key column*/
    sample_t* pk_table =(sample_t*)malloc(sizeof(sample_t) * pk_len);   /* Initialize primary key vector elements in vector */

    generator(fk_table, fk_len, pk_table, pk_len);

    double time = run_on_gpu(fk_table, fk_len, pk_table, pk_len, false);
    printf("%4.2lf\n", time);
    
    return 0;
}

/* init_data - initializes the array */
void init_data(int *data, int n, int cardinality)
{
    int i;
    for (i = 0; i < n; i++)
        //data[i] = random(cardinality);
        data[i] = 1;
}

__global__ void cuda_nlj_thread(int *data, int *vector, int n, int vec_len, int *CountResult) {
  // used to save vector data
    __shared__ int shared[THREAD_NUM];
    __shared__ int rst_shared[THREAD_NUM];
  //  int result = 0;
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    int vec_block_num = (vec_len + THREAD_NUM) / THREAD_NUM;
    //if (vec_block_num == 0) vec_block_num = 1;
    int vec_rem = vec_len;
    
    int j;
    int tid_limit = THREAD_NUM;

    rst_shared[tid] = 0;

    for (j = 0; j < vec_block_num; j++) {
      //if (j == (vec_block_num - 1) && vec_rem != 0)
        //tid_limit = vec_rem;
      tid_limit = (j == (vec_block_num -1)) ? vec_rem : THREAD_NUM;
      vec_rem -= THREAD_NUM;
      // copy vector to shared memory
      if (tid < tid_limit)
        shared[tid] = vector[THREAD_NUM * j + tid]; 

      __syncthreads();

      int i, k;
      // for debug
      int count = 0;
      for (i = bid * THREAD_NUM + tid; i < n; i += BLOCK_NUM * THREAD_NUM) {
        count++;
        for (k = 0; k < tid_limit; k++) {
          //if (data[i] == shared[k]) {
          if (1 == shared[k]) {
              rst_shared[tid] += 1;
          }
        }
      }
    }

    //__syncthreads();

    //rst_shared[tid] = result;

    __syncthreads();

    int i;
    if (tid == 0) {
        for (i = 1; i < THREAD_NUM; i++) {
            rst_shared[0] += rst_shared[i];
        }
        CountResult[bid] = rst_shared[0];
        
        //printf("count result[%d]: %d\n", bid, CountResult[bid]);
    }
}

double run_on_gpu(int *data, int n, int *vector, int vec_len, bool hasTransferTime) {
    int result = 0;
    //int *dev_data, *dev_vector;
    int *results;
    uint64_t timer1, timer2;
    double cycles_per_tuple1 = 0.0, cycles_per_tuple2 = 0.0;
    
    device_args_t deviceArgs[DEVICE_MAX_COUNT];
    int cudaDeviceNum = 0;
    // get number of device
    checkCuda(cudaGetDeviceCount(&cudaDeviceNum));

    //cudaStream_t streams[nstream];
    int i, j;

    int dataBlock = n / cudaDeviceNum; 
    int dataRemSize = n;
    int *tmpPtr = data;

    results = (int*)malloc(sizeof(int) * BLOCK_NUM);

    for (i = 0; i < cudaDeviceNum; i++) {
      int dataSize = (i == (cudaDeviceNum - 1) ? dataRemSize : dataBlock);
      checkCuda(cudaSetDevice(i));

      dataRemSize -= dataSize;
      deviceArgs[i].results = (int*)malloc(sizeof(int) * BLOCK_NUM);
      checkCuda(cudaMalloc((void**)&deviceArgs[i].dev_data, sizeof(int) * dataSize));
      checkCuda(cudaMalloc((void**)&deviceArgs[i].dev_vector, sizeof(int) * vec_len));
    
      for (j = 0; j < nstream; j++) {
        checkCuda(cudaMalloc((void**)&deviceArgs[i].dev_results[j], sizeof(int) * BLOCK_NUM));
      }
    }

    dataRemSize = n;
    cudaStream_t copyStreams[DEVICE_MAX_COUNT];
    for (i = 0; i < cudaDeviceNum; i++) {
      int dataSize = (i == (cudaDeviceNum - 1) ? dataRemSize : dataBlock);
      checkCuda(cudaSetDevice(i));

      dataRemSize -= dataSize;
      checkCuda(cudaStreamCreate(&copyStreams[i]));
      checkCuda(cudaMemcpyAsync(deviceArgs[i].dev_data, tmpPtr, sizeof(int)
            * dataSize, cudaMemcpyHostToDevice, copyStreams[i])); 
      tmpPtr += dataSize;

      // Time for vector transfering and GPU computing
      startTimer(&timer2);
      checkCuda(cudaMemcpyAsync(deviceArgs[i].dev_vector, vector, sizeof(int)
            * vec_len,  cudaMemcpyHostToDevice, copyStreams[i])); 
      stopTimer(&timer2); 
    
      cycles_per_tuple1 += timer2;
    }

    dataRemSize = n;
    startTimer(&timer1);
    for (i = 0; i < cudaDeviceNum; i++) {
      int dataSize = (i == (cudaDeviceNum - 1) ? dataRemSize : dataBlock);
      checkCuda(cudaSetDevice(i));

      int numDatastr = dataSize / nstream;
      int numData = dataSize;

      dataRemSize -= dataSize;
      
      checkCuda(cudaStreamSynchronize(copyStreams[i]));  

      int tmpDataLen = 0;
      for (j = 0; j < nstream; j++) {
        tmpDataLen = (j == (nstream - 1) ? numData : numDatastr);
        printf("tmp data len:%d\n", tmpDataLen);
        numData -= numDatastr;
        checkCuda(cudaStreamCreate(&deviceArgs[i].streams[j]));
        cuda_nlj_thread<<<BLOCK_NUM, THREAD_NUM, 2 * THREAD_NUM * sizeof(int),
          deviceArgs[i].streams[j]>>>(deviceArgs[i].dev_data + j * numDatastr,
              deviceArgs[i].dev_vector, tmpDataLen, vec_len, deviceArgs[i].dev_results[j]);
      }
    }

    for (i = 0; i < cudaDeviceNum; i++) {
      checkCuda(cudaSetDevice(i));
      for (j = 0; j < nstream; j++) {
        checkCuda(cudaStreamSynchronize(deviceArgs[i].streams[j]));
      }
    }
    
    stopTimer(&timer1); 
    cycles_per_tuple2 = (double)timer1 / (double)n;
    cycles_per_tuple1 = (double)cycles_per_tuple2 + ((double)cycles_per_tuple1 / (double)n);
     
    int k;
    for (i = 0; i < cudaDeviceNum; i++) {
      for (j = 0; j < nstream; j++) {
        checkCuda(cudaMemcpy(results, deviceArgs[i].dev_results[j], sizeof(int) * BLOCK_NUM, cudaMemcpyDeviceToHost));
        for (k = 0; k < BLOCK_NUM; k++) {
            result += results[k];
        }
      }
    }

    for (i = 0; i < cudaDeviceNum; i++) {
      checkCuda(cudaSetDevice(i));
      for (j = 0; j < nstream; j++) {
        checkCuda(cudaStreamDestroy(deviceArgs[i].streams[j]));
        checkCuda(cudaFree(deviceArgs[i].dev_results[j]));
      }
      checkCuda(cudaFree(deviceArgs[i].dev_data));
      checkCuda(cudaFree(deviceArgs[i].dev_vector));
    }

    printf("[%d]",result);
    return hasTransferTime ? cycles_per_tuple1 : cycles_per_tuple2;
}
