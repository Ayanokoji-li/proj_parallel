#ifndef PAGERANK_HPP
#define PAGERANK_HPP

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <omp.h>
#include <stdio.h>
#include "cuda_function.hpp"
#define DAMPING_FACTOR 0.85
#define BLOCKSIZE_LIM 256

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
        double tmp = (1 - damping_factor) / num_nodes + damping_factor * sum;
        y[i] = tmp;
        error[i] = (tmp - x[i]);
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
            atomicadd(y+row, tmp);
        }
        __syncthreads();
        y[i] = (1 - damping_factor) / num_nodes + damping_factor * y[i];
        error[i] = (y[i] - x[i]) * (y[i] - x[i]);
    }

    __syncthreads();

    if(i < num_nodes && i != 0)
    {
        atomicadd(error, error[i]);
    }
    else if(i == 0)
    {
        error[0] = sqrt(error[0]);
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
    error[0] = sqrt(error[0]);
}
#endif // PAGERANK_HPP