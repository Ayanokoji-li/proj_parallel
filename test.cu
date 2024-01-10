#include <cuda_runtime.h>
#include <sm_35_atomic_functions.h>
#include <cuda.h>
#include <iostream>
#include <string>
#include <stdio.h>
#include <bitset>
#include "cuda_function.hpp"

int main()
{
    uint64_t array[BLOCK_SIZE];
    for(auto i = 0; i < BLOCK_SIZE; i ++)
    {
        array[i] = i * 10;
    }

    uint64_t *d_arr;
    uint64_t *d_buffer;
    uint64_t *d_res;
    cudaMalloc((void**)&d_arr, sizeof(uint64_t) * 10);
    cudaMalloc((void**)&d_buffer, sizeof(uint64_t) * 10);
    cudaMalloc((void**)&d_res, sizeof(uint64_t));
    cudaMemcpy(d_arr, array, sizeof(uint64_t) * 10, cudaMemcpyHostToDevice);
    cudaMemset(d_res, 0, sizeof(uint64_t));

    work_efficient_scan_kernel<<<1, BLOCK_SIZE>>>(d_arr, d_res, BLOCK_SIZE);
    uint64_t *res = (uint64_t *)malloc(sizeof(uint64_t)*BLOCK_SIZE);
    cudaMemcpy(res, d_res, sizeof(uint64_t)*BLOCK_SIZE, cudaMemcpyDeviceToHost);
    
    for(auto i = 0; i < 10; i ++)
    {
        std::cout << res[i] << std::endl;
    }
    return 0;
}