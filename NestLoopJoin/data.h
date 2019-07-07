#include <stdint.h>

#define MODEL_SIZE 512
#define MODEL_UINT_SIZE (MODEL_SIZE / sizeof(uint32_t))

typedef uint32_t bytea[MODEL_UINT_SIZE]; // model type
typedef int64_t bigint; // rowkey and device_id type

typedef struct {
    bigint rowkey;
    bigint device_id; 
    bytea model;
} sample_t;

typedef struct {
    bigint rowkey;
    int count;
} result_t;

inline __host__ __device__ float similar(bytea a, bytea b) {
    int i;
    int sim = 0;
    for (i = 0; i < MODEL_UINT_SIZE; i++) {
        uint32_t tmp = ~(a[i] ^ b[i]);
        // while (tmp) {
        //     if (tmp & 1)  sim++;
        //     tmp = tmp >> 1;
        // }
        tmp = (tmp & 0x55555555) + ((tmp & 0xaaaaaaaa) >> 1);
        tmp = (tmp & 0x33333333) + ((tmp & 0xcccccccc) >> 2);
        tmp = (tmp & 0x0f0f0f0f) + ((tmp & 0xf0f0f0f0) >> 4);
        tmp = (tmp & 0x00ff00ff) + ((tmp & 0xff00ff00) >> 8);
        tmp = (tmp & 0x0000ffff) + ((tmp & 0xffff0000) >> 16);
        sim += tmp;
    }
    return sim * 0.4 / (512 * 8) + 0.6;
}

inline __host__ __device__ bool predicate(sample_t *t1, sample_t *t2) {
    return t1->device_id > t2->device_id
        && t1->rowkey != t2 -> rowkey
        && similar(t1->model, t2->model) > 0.9;
}