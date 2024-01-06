#include <cuda.h>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <bitset>
#include <omp.h>
#include <cusparse.h>

#define DAMPING_FACTOR 0.85
#define EPSILON 1e-6

// T is the type of the matrix's value
template <typename T>
struct CSRMatrix {
    T* csrVal;
    unsigned long long* csrRowPtr;
    unsigned long long* csrColInd;
    unsigned long long EdgeNum;
    unsigned long long VertexNum;

    CSRMatrix() {};
    CSRMatrix(const std::string &fileName) {
        readFromFile(fileName);
    }

    ~CSRMatrix() {
        free(csrVal);
        free(csrRowPtr);
        free(csrColInd);
    }

    void Transpose(CSRMatrix<T> &dst)
    {
        dst.VertexNum = VertexNum;
        dst.EdgeNum = EdgeNum;
        dst.csrVal = (T*)malloc(EdgeNum * sizeof(T));
        dst.csrRowPtr = (unsigned long long*)malloc((VertexNum + 1) * sizeof(unsigned long long));
        dst.csrColInd = (unsigned long long*)malloc(EdgeNum * sizeof(unsigned long long));
        // cusparseHandle_t handle;
        // cusparseCreate(&handle);

        // void *buffer = NULL;
        // unsigned long long *cscColPtr, *cscRowInd;
        // T *cscVal;

        // unsigned long long bufferSize = 0;
        // cusparseCsr2cscEx2_bufferSize(handle, EdgeNum, EdgeNum, VertexNum, 
        //                                 csrVal, csrRowPtr, csrColInd, dst.csrVal, dst.csrRowPtr, dst.csrColInd, 
        //                                 CUDA_R_64F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, 
        //                                 &bufferSize);
        // buffer = malloc(bufferSize);

        // cusparseCsr2cscEx2(handle, EdgeNum, EdgeNum, VertexNum,
        //                 csrVal, csrRowPtr, csrColInd,
        //                 dst.cscVal, dst.cscRowInd, dst.cscColPtr,
        //                 CUDA_R_64F,
        //                 CUSPARSE_ACTION_NUMERIC,
        //                 CUSPARSE_INDEX_BASE_ZERO, buffer);

        // cusparseDestroy(handle);
    }

    void readFromFile(const std::string &fileName)
    {
        std::ifstream file(fileName);
        std::string line;
        unsigned long long numRows, numCols;

        // Skip header
        do {
            std::getline(file, line);
        } while (line[0] == '%');

        std::stringstream s(line);
        s >> numRows >> numCols >> EdgeNum;
        VertexNum = numRows;
        csrVal = (T*)malloc(EdgeNum * sizeof(T));
        csrRowPtr = (unsigned long long*)malloc((VertexNum + 1) * sizeof(unsigned long long));
        csrColInd = (unsigned long long*)malloc(EdgeNum * sizeof(unsigned long long));
        unsigned long long index = 0;
        for (unsigned long long i = 0; i < EdgeNum; i++) {
            file >> csrRowPtr[i] >> csrColInd[i] >> csrVal[i];
            csrRowPtr[i]--;  // Convert to 0-based index
            csrColInd[i]--;
        }
        csrRowPtr[VertexNum] = EdgeNum;
    }

    void TransitionProb(CSRMatrix<double> &dst)
    {

    }
};

void TransitionProb(CSRMatrix<int> &src, CSRMatrix<double> &dst)
{
    dst.VertexNum = src.VertexNum;
    dst.EdgeNum = src.EdgeNum;
    dst.csrVal = (double*)malloc(dst.EdgeNum * sizeof(double));
    dst.csrRowPtr = (unsigned long long*)malloc((dst.VertexNum + 1) * sizeof(unsigned long long));
    dst.csrColInd = (unsigned long long*)malloc(dst.EdgeNum * sizeof(unsigned long long));
    dst.csrRowPtr[0] = 0;
    for(unsigned long long i = 0; i < dst.VertexNum; i++)
    {
        unsigned long long sum = 0;
        for(unsigned long long j = src.csrRowPtr[i]; j < src.csrRowPtr[i + 1]; j++)
        {
            sum += src.csrVal[j];
        }
        for(unsigned long long j = src.csrRowPtr[i]; j < src.csrRowPtr[i + 1]; j++)
        {
            dst.csrVal[j] = (double)src.csrVal[j] / sum;
            dst.csrColInd[j] = src.csrColInd[j];
        }
        dst.csrRowPtr[i + 1] = src.csrRowPtr[i + 1];
    }
}

// no stl
struct EFGMatrix {
    unsigned long long* data;
    unsigned long long* vlist;
    unsigned long long MaxVal;
    unsigned long long EdgeNum;
    unsigned long long VertexNum;

    EFGMatrix() {};

    ~EFGMatrix() {
        free(data);
        free(vlist);
    }
};

template <typename T>
__global__ void pagerank_csr(T* csrVal, unsigned long long* csrRowPtr, unsigned long long* csrColInd, double* x, double* y, unsigned long long num_nodes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_nodes) {
        double sum = 0.0f;
        for (int j = csrRowPtr[i]; j < csrRowPtr[i + 1]; j++) {
            int col = csrColInd[j];
            sum += x[col] * (double)csrVal[j];
        }
        y[i] = (1 - DAMPING_FACTOR) / num_nodes + DAMPING_FACTOR * sum;
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
    CSRMatrix<double> transition;
    TransitionProb(matrix, transition);
    CSRMatrix<double> tmp;
    transition.Transpose(tmp);

    std::cout << "init end" << std::endl;
    // EFGMatrix efg();
    int N = transition.VertexNum;
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
    double error = 1.0f;
    
    while (error > EPSILON && max_iterations > 0) {
        pagerank_csr<double><<<(N + 255) / 256, 256>>>(d_Val, d_RowPtr, d_ColData, d_x, d_y, N);
        cudaMemcpy(y, d_y, N * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(x, d_x, N * sizeof(double), cudaMemcpyDeviceToHost);
        error = 0.0f;
        for (int i = 0; i < N; i++) {
            error += std::abs(y[i] - x[i]);
        }
        std::swap(d_x, d_y);
        max_iterations--;
    }

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