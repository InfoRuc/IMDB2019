#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <stdint.h>

typedef unsigned int bytea[128];
typedef int bigint;

typedef struct {
  bigint rowkey;
  bigint device_id; 
  bytea model;
} sample_t;

float cpu_similar(bytea a, bytea b) {
  int i;
  int sim = 0;
  for (i = 0; i < 128; i++) {
    unsigned int tmp = ~(a[i] ^ b[i]);
    while (tmp) {
      if (tmp & 1)  sim++;
      tmp = tmp >> 1;
    } 
  }
  return (float) sim * 0.4 / (512.0 * 8) + 0.6;
}

__device__ float gpu_similar(bytea a, bytea b) {
  int i;
  int sim = 0;
  for (i = 0; i < 128; i++) {
    unsigned int tmp = ~(a[i] ^ b[i]);
    while (tmp) {
      if (tmp & 1)  sim++;
      tmp = tmp >> 1;
    } 
  }
  return (float) sim * 0.4 / (512.0 * 8) + 0.6;
}

