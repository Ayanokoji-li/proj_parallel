#ifndef CUDA_FUNCTIONS_H
#define CUDA_FUNCTIONS_H

#include <cuda.h>
#define BLOCK_SIZE 1024
__forceinline__ __device__ void atomicadd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    } while (assumed != old);
}

__forceinline__ __device__
double warpReduceSum(double val) {
  for (unsigned int offset = warpSize/2; offset > 0; offset /= 2) 
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    printf("var = %e\n", val);
  return val;
}

__forceinline__ __device__
int blockReduceSum(double val) {

  static __shared__ int shared[32]; // Shared mem for 32 partial sums
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  val = warpReduceSum(val);     // Each warp performs partial reduction

  if (lane==0) shared[wid]=val; // Write reduced value to shared memory

  __syncthreads();              // Wait for all partial reductions

  //read from shared memory only if that warp existed
  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

  if (wid==0) val = warpReduceSum(val); //Final reduce within first warp

  return val;
}

__global__ void deviceReduceKernel(double* in, double* out, int N)
{
  double sum = 0;
  //reduce multiple elements per thread
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
       i < N; 
       i += blockDim.x * gridDim.x) {
    sum += in[i];
  }
  sum = blockReduceSum(sum);
  if (threadIdx.x==0)
  {
    out[blockIdx.x]=sum;
  }
}

// wrong. always output zero
void deviceReduce(double *in, double* out, uint64_t N) {
  uint64_t threads = 512;
  uint64_t blocks = min((N + threads - 1) / threads, 1024LU);
  deviceReduceKernel<<<blocks, threads>>>(in, out, N);
  deviceReduceKernel<<<1, 1024>>>(out, out, blocks);
}

__global__ void prefix_sum(uint64_t * array)
{
    uint64_t stride = 1;
    while(stride <= blockDim.x)
    {
        uint64_t index = 2 * stride * (threadIdx.x + 1) - 1;
        if(index < 2 * blockDim.x)
        {
            array[index] += array[index - stride];
        }
        stride *= 2;
        __syncthreads();
    }

    stride = blockDim.x / 2;
    while(stride > 0)
    {
        uint64_t index = 2 * stride * (threadIdx.x + 1) - 1;
        if(index + stride < 2 * blockDim.x)
        {
            array[index + stride] += array[index];
        }
        stride /= 2;
        __syncthreads();
    }
}

__global__ void work_efficient_scan_kernel(uint64_t *X, uint64_t *Y, uint64_t InputSize) {
// XY[2*BLOCK_SIZE] is in shared memory
  __shared__ uint64_t XY[BLOCK_SIZE * 2];
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < InputSize) {XY[threadIdx.x] = X[i];}
  // the code below performs iterative scan on XY　　
  for (unsigned int stride = 1; stride <= BLOCK_SIZE; stride *= 2) {
    __syncthreads();
  int index = (threadIdx.x+1)*stride*2 - 1; 
    if(index < 2*BLOCK_SIZE)
        XY[index] += XY[index - stride];//index is alway bigger than stride
    __syncthreads();
  }
  
  for (unsigned int stride = BLOCK_SIZE/2; stride > 0 ; stride /= 2) {
    __syncthreads();
    int index = (threadIdx.x+1)*stride*2 - 1;
    if(index < 2*BLOCK_SIZE)
        XY[index + stride] += XY[index];
  }
  __syncthreads();
  if (i < InputSize) Y[i] = XY[threadIdx.x];
}

#endif