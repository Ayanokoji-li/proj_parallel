#include <cuda.h>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <bitset>
#include <omp.h>
#include <tuple>
#include "CSR.hpp"

#define DAMPING_FACTOR 0.85
#define EPSILON 1e-6

// no stl
template <typename T>
struct EFGMatrix {
    T* efgVal;
    unsigned long long* efgRowPtr;
    std::bitset* efgLowBits;
    int* efgLowBitsNum;
    std::bitset* efgHighBits;
    unsigned long long EdgeNum;
    unsigned long long VertexNum;

    EFGMatrix() {};

    ~EFGMatrix() {
        free(data);
        free(vlist);
    }
};

template <typename T>
__global__ void pagerank_csr(T* csrVal, unsigned long long* csrRowPtr, unsigned long long* csrColInd, double* x, double* y, double* error, unsigned long long num_nodes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_nodes) {
        double sum = 0.0f;
        for (int j = csrRowPtr[i]; j < csrRowPtr[i + 1]; j++) {
            int col = csrColInd[j];
            sum += x[col] * (double)csrVal[j];
        }
        y[i] = (1 - DAMPING_FACTOR) / num_nodes + DAMPING_FACTOR * sum;
        error[i] = (y[i] - x[i]) * (y[i] - x[i]);
    }

    __syncthreads();
    // reduction error
    int stride = 1;
    while (stride < blockDim.x) {
        int index = 2 * stride * i;
        if (index < blockDim.x) {
            error[index] += error[index + stride];
        }
        stride *= 2;
        __syncthreads();
    }
    if (i == 0) {
        error[0] = sqrt(error[0]);
    }
}

template <typename T>
void PageRank_cpu(T* csrVal, unsigned long long* csrRowPtr, unsigned long long* csrColInd, double* x, double* y, double* error, unsigned long long num_nodes) {
    for (int i = 0; i < num_nodes; i++) {
        double sum = 0.0f;
        for (int j = csrRowPtr[i]; j < csrRowPtr[i + 1]; j++) {
            int col = csrColInd[j];
            sum += x[col] * (double)csrVal[j];
        }
        y[i] = (1 - DAMPING_FACTOR) / num_nodes + DAMPING_FACTOR * sum;
        error[i] = (y[i] - x[i]) * (y[i] - x[i]);
    }
}


int main(int argc, char** argv) {
    std::string file_name = "web-Google.mtx";
    bool method = 0;
    if(argc != 2) 
    {
        std::cout << "default into file: web-Google.mtx" << std::endl;
    }
    else
    {
        file_name = argv[1];
        std::cout << "input file: " << file_name << std::endl;
    }

    // Initialize host value
    CSRMatrix<int> matrix(file_name);
    CSRMatrix<double> tmp;
    TransitionProb(matrix, tmp);
    CSRMatrix<double> transition;
    tmp.Transpose(transition);

    std::cout << "init end" << std::endl;
    // EFGMatrix efg();
    unsigned long long N = transition.VertexNum;
    double* x = (double*)malloc(N * sizeof(double));
    double* y = (double*)malloc(N * sizeof(double));
    double init = 1.0f / N;
    for(int i = 0; i < N; i++) {
        x[i] = init;
    }
    memset(y, 0, N * sizeof(double));

    // Initialize device value
    double* d_Val;
    unsigned long long* d_RowPtr;
    unsigned long long* d_ColData;
    double* d_x;
    double* d_y;
    if(method == 0)
    {
        cudaMalloc((void**)&d_Val, transition.EdgeNum * sizeof(double));
        cudaMalloc((void**)&d_RowPtr, (transition.VertexNum + 1) * sizeof(unsigned long long));
        cudaMalloc((void**)&d_ColData, transition.EdgeNum * sizeof(unsigned long long));
        cudaMemcpy(d_Val, transition.csrVal, transition.EdgeNum * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_RowPtr, transition.csrRowPtr, (transition.VertexNum + 1) * sizeof(unsigned long long), cudaMemcpyHostToDevice);
        cudaMemcpy(d_ColData, transition.csrColInd, transition.EdgeNum * sizeof(unsigned long long), cudaMemcpyHostToDevice);
    }
    else
    {
        // cudaMalloc((void**)&d_Val, efg.data.size() * sizeof(unsigned long long));
        // cudaMalloc((void**)&d_RowPtr, (efg.vlist.size()) * sizeof(unsigned long long));
        // cudaMalloc((void**)&d_ColData, efg.data.size() * sizeof(unsigned long long));
        // cudaMemcpy(d_Val, efg.data, efg.data.size() * sizeof(unsigned long long), cudaMemcpyHostToDevice);
        // cudaMemcpy(d_RowPtr, efg.vlist, (efg.vlist.size()) * sizeof(unsigned long long), cudaMemcpyHostToDevice);
        // cudaMemcpy(d_ColData, efg.data, efg.data.size() * sizeof(unsigned long long), cudaMemcpyHostToDevice);
    }
    cudaMalloc((void**)&d_x, N * sizeof(double));
    cudaMalloc((void**)&d_y, N * sizeof(double));
    cudaMemcpy(d_x, x, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N * sizeof(double), cudaMemcpyHostToDevice);

    // Perform PageRank iterations
    int max_iterations = 1000;
    double *error = (double*)malloc(N * sizeof(double));
    error[0] = 1.0f;

    double *d_error;
    cudaMalloc((void**)&d_error, N * sizeof(double));
    
    while (error[0] > EPSILON && max_iterations > 0) {
        pagerank_csr<double><<<(N + 255) / 256, 256>>>(d_Val, d_RowPtr, d_ColData, d_x, d_y,d_error, N);
        cudaMemcpy(error, d_error,sizeof(double), cudaMemcpyDeviceToHost);
        std::swap(d_x, d_y);
        max_iterations--;
    }
    cudaMemcpy(y, d_y, N * sizeof(double), cudaMemcpyDeviceToHost);

    // Free memory on the device
    cudaFree(d_RowPtr);
    cudaFree(d_ColData);
    cudaFree(d_x);
    cudaFree(d_y);

    // Free memory on the host
    free(x);
    free(y);

    return 0;
}