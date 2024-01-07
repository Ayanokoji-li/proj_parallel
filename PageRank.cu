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

#define INTBITS 32
#define GET_LOW_BIT(x, n) ((x) & ((1 << (n)) - 1))
#define GET_HIGH_BIT(x, n) ((x) >> (n))
static const char ZERO[INTBITS] = {0};

// no stl
struct EFGMatrix {
    double* efgVal;
    uint64_t* efgRowPtr;
    void** efgLowBits;
    int* efgLowBitsNum;
    void** efgHighBits;
    int* efghighBitsNum;
    uint64_t EdgeNum;
    uint64_t VertexNum;

    EFGMatrix(CSRMatrix &csr) 
    {
        malloc(efgVal, csr.EdgeNum * sizeof(double));
        malloc(efgRowPtr, (csr.VertexNum + 1) * sizeof(uint64_t));
        memcpy(efgVal, csr.csrVal, csr.EdgeNum * sizeof(double));
        memcpy(efgRowPtr, csr.csrRowPtr, (csr.VertexNum + 1) * sizeof(uint64_t));
        EdgeNum = csr.EdgeNum;
        VertexNum = csr.VertexNum;
        efgLowBits = (void**)malloc((VertexNum + 1) * sizeof(void*));
        efgLowBitsNum = (int*)malloc((VertexNum) * sizeof(int)); // low bit num per neighbors
        efgHighBits = (void**)malloc((VertexNum + 1) * sizeof(void*));
        efgHighBitsNum = (int*)malloc((VertexNum) * sizeof(int)); // high bit num total
        for(uint64_t i = 0; i < VertexNum + 1; i++) {
            if(auto num_neighbors = efgRowPtr[i+1] - efgRowPtr[i]; num_neighbors != 0)
            {
                efgLowBitsNum[i] = std::floor(std::log2(csr.csrColInd[efgRowPtr[i+1]-1]));
                if(efgLowBitsNum[i] < 0) {
                    efgLowBitsNum[i] = 0;
                }
                else
                {
                    if(efgLowBitsNum[i] % 2 != 0) {
                        efgLowBitsNum[i] = (efgLowBitsNum[i] / 2 + 1) * 2;
                    }
                }
                efgLowBits[i] = malloc((num_neighbors * efgLowBitsNum[i] - 1) / 8 + 1);

                efgHighBitsNum[i] = csr.csrColInd[efgRowPtr[i+1]-1] >> efgLowBitsNum[i] + num_neighbors;
                if(efgHighBitsNum[i] < 0) {
                    efgHighBitsNum[i] = 0;
                }
                else
                {
                    if(efghighBitsNum % 4 != 0) {
                        efghighBitsNum = (efghighBitsNum / 4 + 1) * 4;
                    }
                }
                efgHighBits[i] = malloc((efgHighBitsNum[i]-1) / 8 + 1);
                memset(efgLowBits[i], 0, (efgLowBitsNum[i] - 1) / 8 + 1);
                uint64_t prevHighBits = 0;
                uint64_t pos = 0;
                for(auto j = efgRowPtr[i]; j < efgRowPtr[i+1]; j++) {
                    auto num = j - efgRowPtr[i];
                    uint64_t lowBits = GET_LOW_BIT(csr.csrColInd[j], efgLowBitsNum[i]);
                    efgLowBits[i + num* efgLowBitsNum[i]/8] ^= lowBits << (8 - num * efgLowBitsNum[i] % 8);
                    uint64_t highBits = GET_HIGH_BIT(csr.csrColInd[j], efgLowBitsNum[i]);
                    uint64_t diff = highBits - prevHighBits;
                    prevHighBits = highBits;
                    pos += diff + 1;
                    efgHighBits[i + pos/8] ^= 1 << (8 - pos % 8);
                }
            }
        }
    };

    ~EFGMatrix() {
        free(efgVal);
        free(efgRowPtr);
        free(efgLowBits);
        free(efgLowBitsNum);
        free(efgHighBits);
    }
};

struct cuEFGMatrix
{
    double* d_efgVal;
    uint64_t* d_efgRowPtr;
    void* d_efgLowBits;
    int* d_efgLowBitsNum;
    void* d_efgHighBits;
    int* d_efghighBitsNum;
    uint64_t d_EdgeNum;
    uint64_t d_VertexNum;

    cuEFGMatrix(EFGMatrix &efg) 
    {
        cudaMalloc((void**)&d_efgVal, efg.EdgeNum * sizeof(double));
        cudaMalloc((void**)&d_efgRowPtr, (efg.VertexNum + 1) * sizeof(uint64_t));
        cudaMemcpy(d_efgVal, efg.efgVal, efg.EdgeNum * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_efgRowPtr, efg.efgRowPtr, (efg.VertexNum + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice);
        int efgLowBitsNum[efg.VertexNum];
        int efghighBitsNum[efg.VertexNum];

        //do prefix sum of efgLowBitsNum
        for(int i = 0; i < efg.VertexNum; i++) {
            if(i == 0)
            {
                efgLowBitsNum[i] = efg.efgLowBitsNum[i] * (efg.efgRowPtr[i+1] - efg.efgRowPtr[i]);
                efghighBitsNum[i] = efg.efghighBitsNum[i];
            }
            else
            {
                efgLowBitsNum[i] = efg.efgLowBitsNum[i] * (efg.efgRowPtr[i+1] - efg.efgRowPtr[i]) + efgLowBitsNum[i-1];
                efghighBitsNum[i] = efg.efghighBitsNum[i] + efghighBitsNum[i-1];
            }
        }
        cudaMalloc((void**)&d_efgLowBits, efgLowBitsNum[efg.VertexNum-1] * sizeof(void*));
        
        

    }

    ~cuEFGMatrix() {
        cudaFree(efgVal);
        cudaFree(efgRowPtr);
        cudaFree(efgLowBits);
        cudaFree(efgLowBitsNum);
        cudaFree(efgHighBits);
        cudaFree(efghighBitsNum);
    }
};
}

template <typename T>
__global__ void pagerank_csr(T* csrVal, uint64_t* csrRowPtr, uint64_t* csrColInd, double* x, double* y, double* error, uint64_t num_nodes) {
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
void PageRank_cpu(T* csrVal, uint64_t* csrRowPtr, uint64_t* csrColInd, double* x, double* y, double* error, uint64_t num_nodes) {
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
    CSRMatrix matrix(file_name);
    CSRMatrix transition{};
    TransitionProb(matrix, transition);

    std::cout << "init end" << std::endl;
    // EFGMatrix efg();
    uint64_t N = transition.VertexNum;
    double* x = (double*)malloc(N * sizeof(double));
    double* y = (double*)malloc(N * sizeof(double));
    double init = 1.0f / N;
    for(int i = 0; i < N; i++) {
        x[i] = init;
    }
    memset(y, 0, N * sizeof(double));

    // Initialize device value
    double* d_Val;
    uint64_t* d_RowPtr;
    uint64_t* d_ColData;
    double* d_x;
    double* d_y;
    if(method == 0)
    {
        cudaMalloc((void**)&d_Val, transition.EdgeNum * sizeof(double));
        cudaMalloc((void**)&d_RowPtr, (transition.VertexNum + 1) * sizeof(uint64_t));
        cudaMalloc((void**)&d_ColData, transition.EdgeNum * sizeof(uint64_t));
        cudaMemcpy(d_Val, transition.csrVal, transition.EdgeNum * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_RowPtr, transition.csrRowPtr, (transition.VertexNum + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_ColData, transition.csrColInd, transition.EdgeNum * sizeof(uint64_t), cudaMemcpyHostToDevice);
    }
    else
    {
        // cudaMalloc((void**)&d_Val, efg.data.size() * sizeof(uint64_t));
        // cudaMalloc((void**)&d_RowPtr, (efg.vlist.size()) * sizeof(uint64_t));
        // cudaMalloc((void**)&d_ColData, efg.data.size() * sizeof(uint64_t));
        // cudaMemcpy(d_Val, efg.data, efg.data.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);
        // cudaMemcpy(d_RowPtr, efg.vlist, (efg.vlist.size()) * sizeof(uint64_t), cudaMemcpyHostToDevice);
        // cudaMemcpy(d_ColData, efg.data, efg.data.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);
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