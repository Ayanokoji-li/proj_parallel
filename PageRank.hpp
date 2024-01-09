#ifndef PAGERANK_HPP
#define PAGERANK_HPP

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <omp.h>
#include <stdio.h>
#define DAMPING_FACTOR 0.85
#define BLOCKSIZE_LIM 256

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600 

#else 

   __device__ double atomicAdd(double* a, double b) 

   { 

      return b; 

   } 

#endif

enum class MATRIX_TYPE {
    CSR,
    CSC
};

enum class COM_TYPE {
    CPU,
    GPU
};

__global__ void PageRank_cuda_csr(double* csrVal, uint64_t* csrRowPtr, uint64_t* csrColInd, double* x, double* y, double* error, uint64_t num_nodes, double damping_factor = DAMPING_FACTOR) {
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_nodes) {
        double sum = 0.0f;
        for (auto j = csrRowPtr[i]; j < csrRowPtr[i + 1]; j++) {
            auto col = csrColInd[j];
            sum += x[col] * (double)csrVal[j];
        }
        y[i] = (1 - damping_factor) / num_nodes + damping_factor * sum;
        error[i] = (y[i] - x[i]) * (y[i] - x[i]);
        atomicAdd(&error[0], error[i]);
    }

    __syncthreads();
    // reduction error
    // int stride = 1;
    // while (stride < num_nodes) {
    //     int index = 2 * stride * i;
    //     if (index < num_nodes) {
    //         error[index] += error[index - stride];
    //     }
    //     stride *= 2;
    //     __syncthreads();
    // }

    if (i == 0) {
        error[0] = sqrt(error[0]);
    }
}

__global__ void PageRank_cuda_csc(double *cscVal, uint64_t *cscColPtr, uint64_t *cscRowInd, double *x, double *y, double *error, uint64_t num_nodes, double damping_factor = DAMPING_FACTOR)
{
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_nodes)
    {
        for (auto j = cscColPtr[i]; j < cscColPtr[i + 1]; j++)
        {
            auto row = cscRowInd[j];
            double tmp = x[row] * cscVal[j];
            y[row] += tmp;
            printf("y[%d] = %e\n", row, y[row]);
        }
        __syncthreads();
        y[i] = (1 - damping_factor) / num_nodes + damping_factor * y[i];
        printf("y[%d] = %e\n", i, y[i]);
        error[i] = (y[i] - x[i]) * (y[i] - x[i]);
    }
}

void PageRank_cpu_csr(double* csrVal, uint64_t* csrRowPtr, uint64_t* csrColInd, double* x, double* y, double* error, uint64_t num_nodes, double damping_factor = DAMPING_FACTOR) {
    #pragma omp parallel for
    for (int i = 0; i < num_nodes; i++) {
        double sum = 0.0f;
        for (int j = csrRowPtr[i]; j < csrRowPtr[i + 1]; j++) {
            int col = csrColInd[j];
            sum += x[col] * (double)csrVal[j];
        }
        y[i] = (1 - damping_factor) / num_nodes + damping_factor * sum;
        error[i] = (y[i] - x[i]) * (y[i] - x[i]);
    }

    //reduction error
    #pragma omp parallel for
    for(auto i = 1; i < num_nodes; i ++) {
        #pragma omp atomic
        error[0] += error[i];
    }
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

#endif // PAGERANK_HPP