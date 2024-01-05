#include <cuda_runtime.h>
#include <stdio.h>

#define N 1000
#define DAMPING_FACTOR 0.85
#define EPSILON 1e-6

__global__ void pagerank(float* A, float* x, float* y, int* dangling_nodes, int num_nodes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_nodes) {
        float sum = 0.0f;
        for (int j = 0; j < num_nodes; j++) {
            if (A[j * num_nodes + i] > 0) {
                sum += x[j] / A[j * num_nodes + i];
            }
        }
        y[i] = (1 - DAMPING_FACTOR) / num_nodes + DAMPING_FACTOR * sum;
        if (dangling_nodes[i] == 1) {
            y[i] += (1 - DAMPING_FACTOR) / num_nodes;
        }
    }
}

__global__ void pagerank(float* csrVal, int* csrRowPtr, int* csrColInd, float* x, float* y, int* dangling_nodes, int num_nodes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_nodes) {
        float sum = 0.0f;
        for (int j = csrRowPtr[i]; j < csrRowPtr[i + 1]; j++) {
            int col = csrColInd[j];
            sum += x[col] / csrVal[j];
        }
        y[i] = (1 - DAMPING_FACTOR) / num_nodes + DAMPING_FACTOR * sum;
        if (dangling_nodes[i] == 1) {
            y[i] += (1 - DAMPING_FACTOR) / num_nodes;
        }
    }
}

int main() {
    float* A, *x, *y;
    int* dangling_nodes;
    float* d_A, *d_x, *d_y;
    int* d_dangling_nodes;

    // Allocate memory on the host
    A = (float*)malloc(N * N * sizeof(float));
    x = (float*)malloc(N * sizeof(float));
    y = (float*)malloc(N * sizeof(float));
    dangling_nodes = (int*)malloc(N * sizeof(int));

    // Initialize A, x, and dangling_nodes

    // Allocate memory on the device
    cudaMalloc((void**)&d_A, N * N * sizeof(float));
    cudaMalloc((void**)&d_x, N * sizeof(float));
    cudaMalloc((void**)&d_y, N * sizeof(float));
    cudaMalloc((void**)&d_dangling_nodes, N * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_A, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dangling_nodes, dangling_nodes, N * sizeof(int), cudaMemcpyHostToDevice);

    // Perform PageRank iterations
    int max_iterations = 100;
    float error = 1.0f;
    while (error > EPSILON && max_iterations > 0) {
        pagerank<<<(N + 255) / 256, 256>>>(d_A, d_x, d_y, d_dangling_nodes, N);
        cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
        error = 0.0f;
        for (int i = 0; i < N; i++) {
            error += fabs(y[i] - x[i]);
            x[i] = y[i];
        }
        max_iterations--;
    }

    // Free memory on the device
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_dangling_nodes);

    // Free memory on the host
    free(A);
    free(x);
    free(y);
    free(dangling_nodes);

    return 0;
}
