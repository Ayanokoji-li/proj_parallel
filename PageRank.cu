#include <cuda.h>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <bitset>

#define DAMPING_FACTOR 0.85
#define EPSILON 1e-6


// suppose weight is integer
struct CSRMatrix {
    unsigned long long* csrVal;
    unsigned long long* csrRowPtr;
    unsigned long long* csrColInd;
    unsigned long long MaxVal;
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

    // input file format: Matrix market format
    void readFromFile(const std::string &fileName) {
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
        csrRowPtr = (unsigned long long*)malloc((numRows + 1) * sizeof(unsigned long long));
        csrVal = (unsigned long long*)malloc(EdgeNum * sizeof(unsigned long long));
        csrColInd = (unsigned long long*)malloc(EdgeNum * sizeof(unsigned long long));

        unsigned long long row, col;
        unsigned long long val;
        for (int i = 0; i < EdgeNum; i++) {
            file >> row >> col >> val;
            row--;  // Convert to 0-based index
            col--;
            csrVal[i] = val;
            MaxVal = std::max(MaxVal, val);
            csrColInd[i] = col;
            csrRowPtr[row + 1]++;
        }

        // Compute row pointer array
        for (unsigned long long i = 1; i <= numRows; i++) {
            csrRowPtr[i] += csrRowPtr[i - 1];
        }
    }
};

// no stl
struct EFGMatrix {
    unsigned long long* data;
    unsigned long long* vlist;
    unsigned long long MaxVal;
    unsigned long long EdgeNum;
    unsigned long long VertexNum;

    EFGMatrix() {};
    EFGMatrix(const std::string &fileName) {
        readFromFile(fileName);
    }

    ~EFGMatrix() {
        free(data);
        free(vlist);
    }

    // input file format: Matrix market format
    void readFromFile(const std::string &fileName) {
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
        data = (unsigned long long*)malloc(EdgeNum * sizeof(unsigned long long));
        vlist = (unsigned long long*)malloc((numRows + 1) * sizeof(unsigned long long));

        unsigned long long row, col;
        unsigned long long val;
        for (int i = 0; i < EdgeNum; i++) {
            file >> row >> col >> val;
            row--;  // Convert to 0-based index
            col--;
            data[i] = val;
            MaxVal = std::max(MaxVal, val);
            vlist[row + 1]++;
        }

        // Compute row pointer array
        for (unsigned long long i = 1; i <= numRows; i++) {
            vlist[i] += vlist[i - 1];
        }
    }
};

__global__ void pagerank(unsigned long long* csrVal, unsigned long long* csrRowPtr, unsigned long long* csrColInd, double* x, double* y, unsigned long long num_nodes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_nodes) {
        double sum = 0.0f;
        unsigned long long num_neighbors = csrRowPtr[i + 1] - csrRowPtr[i];
        for (int j = csrRowPtr[i]; j < csrRowPtr[i + 1]; j++) {
            int col = csrColInd[j];
            sum += x[col] / csrVal[j];
        }
        y[i] = (1 - DAMPING_FACTOR) / num_nodes + DAMPING_FACTOR * sum;
        if (num_neighbors == 0) {
            y[i] += (1 - DAMPING_FACTOR) / num_nodes;
        }
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
    CSRMatrix csr(file_name);
    EFGMatrix efg();
    int N = csr.VertexNum;
    double* x = (double*)malloc(N * sizeof(double));
    double* y = (double*)malloc(N * sizeof(double));
    double init = 1.0f / N;
    for(int i = 0; i < N; i++) {
        x[i] = init;
    }
    memset(y, 0, N * sizeof(double));

    // Initialize device value
    unsigned long long* d_Val;
    unsigned long long* d_RowPtr;
    unsigned long long* d_ColData;
    double* d_x;
    double* d_y;
    if(method == 0)
    {
        cudaMalloc((void**)&d_Val, csr.EdgeNum * sizeof(unsigned long long));
        cudaMalloc((void**)&d_RowPtr, (csr.VertexNum + 1) * sizeof(unsigned long long));
        cudaMalloc((void**)&d_ColData, csr.EdgeNum * sizeof(unsigned long long));
        cudaMemcpy(d_Val, csr.csrVal, csr.EdgeNum * sizeof(unsigned long long), cudaMemcpyHostToDevice);
        cudaMemcpy(d_RowPtr, csr.csrRowPtr, (csr.VertexNum + 1) * sizeof(unsigned long long), cudaMemcpyHostToDevice);
        cudaMemcpy(d_ColData, csr.csrColInd, csr.EdgeNum * sizeof(unsigned long long), cudaMemcpyHostToDevice);
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
    double* temp;
    while (error > EPSILON && max_iterations > 0) {
        pagerank<<<(N + 255) / 256, 256>>>(d_Val, d_RowPtr, d_ColData, d_x, d_y, N);
        cudaMemcpy(y, d_y, N * sizeof(double), cudaMemcpyDeviceToHost);
        error = 0.0f;
        for (int i = 0; i < N; i++) {
            error += std::abs(y[i] - x[i]);
        }
        std::swap(x, y);
        max_iterations--;
    }

    // CUP pagerank
    double* c_x = (double*)malloc(N * sizeof(double));
    double* c_y = (double*)malloc(N * sizeof(double));
    for(int i = 0; i < N; i++) {
        c_x[i] = init;
    }
    memset(c_y, 0, N * sizeof(double));
    while (error > EPSILON && max_iterations > 0) {
        for (int i = 0; i < N; i++) {
            double sum = 0.0f;
            unsigned long long num_neighbors = csr.csrRowPtr[i + 1] - csr.csrRowPtr[i];
            for (int j = csr.csrRowPtr[i]; j < csr.csrRowPtr[i + 1]; j++) {
                int col = csr.csrColInd[j];
                sum += c_x[col] / csr.csrVal[j];
            }
            c_y[i] = (1 - DAMPING_FACTOR) / N + DAMPING_FACTOR * sum;
            if (num_neighbors == 0) {
                c_y[i] += (1 - DAMPING_FACTOR) / N;
            }
        }
        error = 0.0f;
        for (int i = 0; i < N; i++) {
            error += std::abs(c_y[i] - c_x[i]);
        }
        std::swap(c_x, c_y);
        max_iterations--;
    }

    unsigned long long i;
    // verify result between GPU and CPU
    for(i = 0; i < N; i ++)
    {
        if(std::abs(c_y[i] - y[i]) > 1e-4)
        {
            std::cout << "error in " << i << " " << c_y[i] << " " << y[i] << std::endl;
            break;
        }
    }
    if(i == N)
    {
        std::cout << "verify success" << std::endl;
    }

    // Free memory on the device
    cudaFree(d_Val);
    cudaFree(d_RowPtr);
    cudaFree(d_ColData);
    cudaFree(d_x);
    cudaFree(d_y);

    // Free memory on the host
    free(x);
    free(y);

    return 0;
}