#include <cusparse_v2.h>
#include <iostream>
#include <vector>
// #include "mmio.h"

#define DAMPING_FACTOR 0.85
#define MAX_ITER 100
#define TOLERANCE 1e-6

__global__ void pagerank_kernel(int num_nodes, int* csrRowPtr, int* csrColInd, double* x, double* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_nodes) {
        double sum = 0.0;
        int start = csrRowPtr[i];
        int end = csrRowPtr[i + 1];
        for (int j = start; j < end; j++) {
            int col = csrColInd[j];
            sum += x[col] / (end - start);
        }
        y[i] = (1 - DAMPING_FACTOR) / num_nodes + DAMPING_FACTOR * sum;
    }
}

void pagerank(int num_nodes, int* csrRowPtr, int* csrColInd) {
    double* x;
    double* y;
    cudaMallocManaged(&x, num_nodes * sizeof(double));
    cudaMallocManaged(&y, num_nodes * sizeof(double));

    for (int i = 0; i < num_nodes; i++) {
        x[i] = 1.0 / num_nodes;
    }

    for (int iter = 0; iter < MAX_ITER; iter++) {
        pagerank_kernel<<<(num_nodes + 255) / 256, 256>>>(num_nodes, csrRowPtr, csrColInd, x, y);
        cudaDeviceSynchronize();

        double error = 0.0;
        for (int i = 0; i < num_nodes; i++) {
            error += abs(y[i] - x[i]);
        }

        if (error < TOLERANCE) {
            break;
        }

        std::swap(x, y);
    }

    for (int i = 0; i < num_nodes; i++) {
        std::cout << "Node " << i << ": PageRank = " << x[i] << std::endl;
    }

    cudaFree(x);
    cudaFree(y);
}

int main() {
    int num_nodes;
    int num_edges;
    int* csrRowPtr;
    int* csrColInd;

    // Read the graph from the web-Google.mtx file in matrix market format
    // You need to implement this function
    read_matrix_market("web-Google.mtx", &num_nodes, &num_edges, &csrRowPtr, &csrColInd);

    // Calculate the PageRank
    pagerank(num_nodes, csrRowPtr, csrColInd);

    return 0;
}