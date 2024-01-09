#ifndef CSR_HPP
#define CSR_HPP

#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <omp.h>


using COO = std::vector<std::tuple<uint64_t, uint64_t, double>>;

struct comp_1 {
    bool operator()(const std::tuple<uint64_t, uint64_t, double> &a, const std::tuple<uint64_t, uint64_t, double> &b) {
        return std::get<0>(a) < std::get<0>(b);
    }
};


struct comp_2 {
    bool operator()(const std::tuple<uint64_t, uint64_t, double> &a, const std::tuple<uint64_t, uint64_t, double> &b) {
        return std::get<1>(a) < std::get<1>(b);
    }
};


// double is the type of the matrix's value

struct CSRMatrix {
    double* csrVal = nullptr;
    uint64_t* csrRowPtr = nullptr;
    uint64_t* csrColInd = nullptr;
    uint64_t EdgeNum;
    uint64_t VertexNum;
    bool isCSC = false;

    CSRMatrix() {};
    CSRMatrix(const std::string &fileName) {
        readFromFile(fileName);
    }

    ~CSRMatrix() {
        free(csrVal);
        free(csrRowPtr);
        free(csrColInd);
    };

    void swap(CSRMatrix &other)
    {
        std::swap(csrVal, other.csrVal);
        std::swap(csrRowPtr, other.csrRowPtr);
        std::swap(csrColInd, other.csrColInd);
        std::swap(EdgeNum, other.EdgeNum);
        std::swap(VertexNum, other.VertexNum);
    }

    void Transpose(CSRMatrix &dst)
    {
        dst.VertexNum = VertexNum;
        dst.EdgeNum = EdgeNum;
        dst.csrVal = (double*)malloc(EdgeNum * sizeof(double));
        dst.csrRowPtr = (uint64_t*)malloc((VertexNum + 1) * sizeof(uint64_t));
        dst.csrColInd = (uint64_t*)malloc(EdgeNum * sizeof(uint64_t));
        memset(dst.csrRowPtr, 0, (VertexNum + 1) * sizeof(uint64_t));
        memset(dst.csrColInd, 0, (VertexNum + 1) * sizeof(double));

        // col_index, row_index
        COO coo(EdgeNum);
        std::vector<uint64_t> col_num(VertexNum, 0);
        #pragma omp parallel for
        for(uint64_t i = 0; i < VertexNum; i++)
        {
            for(uint64_t j = csrRowPtr[i]; j < csrRowPtr[i + 1]; j++)
            {
                coo[j] = std::make_tuple(csrColInd[j], i, csrVal[j]);
                #pragma omp atomic
                col_num[csrColInd[j]]++;
            }
        }
        
        // prefix sum of col_num with omp
        uint64_t stride = 1;
        while(stride < VertexNum)
        {
            #pragma omp parallel for
            for(uint64_t i = stride; i < VertexNum; i += stride * 2)
            {
                col_num[i] += col_num[i - stride];
            }
            stride *= 2;
        }
        stride = (VertexNum) / 2;
        while (stride > 0)
        {
            #pragma omp parallel for
            for(uint64_t i = stride; i < VertexNum; i += stride * 2)
            {
                col_num[i + stride] += col_num[i];
            }
            stride /= 2;
        }

        #pragma omp parallel for
        for(uint64_t i = 0; i < VertexNum; i++)
        {
            for(uint64_t j = csrRowPtr[i]; j < csrRowPtr[i + 1]; j++)
            {
                uint64_t pos;
                #pragma omp critical
                {
                    pos = col_num[csrColInd[j]] ++;
                }
                dst.csrVal[pos] = csrVal[j];
                dst.csrColInd[pos] = i;
            }
        }

        // std::stable_sort(coo.begin(), coo.end(), comp_1());

        // std::vector<uint64_t> row_num(VertexNum+1, 0);
        // std::vector<uint64_t> row_num_tmp(VertexNum+1, 0);
        // #pragma omp parallel for
        // for(uint64_t i = 0; i < EdgeNum; i++)
        // {
        //     row_num[std::get<0>(coo[i]) + 1]++;
        // }

        // dst.csrRowPtr[0] = 0;
        // for(uint64_t i = 0; i < EdgeNum; i++)
        // {
        //     dst.csrVal[i] = std::get<2>(coo[i]);
        //     dst.csrColInd[i] = std::get<1>(coo[i]);
        //     dst.csrRowPtr[std::get<0>(coo[i]) + 1]++;
        // }
        // for(uint64_t i = 1; i <= VertexNum; i++)
        // {
        //     dst.csrRowPtr[i] += dst.csrRowPtr[i - 1];
        // }
        auto end = omp_get_wtime();
        std::cout << "Transpose speed: " << GET_GTEPS(end-start, EdgeNum) << " GTEPS" << std::endl;
        std::cout << "Transpose time " << end-start << " s" << std::endl;
    }

    void readFromFile(const std::string &fileName)
    {
        std::ifstream file(fileName);
        std::string line;
        uint64_t numRows, numCols;

        auto start = omp_get_wtime();
        std::cout << "Read file start" << std::endl;
        // Skip header
        do {
            std::getline(file, line);
        } while (line[0] == '%');

        std::stringstream s(line);
        s >> numRows >> numCols >> EdgeNum;
        VertexNum = numRows;
        csrRowPtr = (uint64_t*)malloc((numRows + 1) * sizeof(uint64_t));
        csrVal = (double*)malloc(EdgeNum * sizeof(double));
        csrColInd = (uint64_t*)malloc(EdgeNum * sizeof(uint64_t));

        uint64_t row, col;
        uint64_t val;
        for (uint64_t i = 0; i < EdgeNum; i++) {
            file >> row >> col >> val;
            row--;  // Convert to 0-based index
            col--;
            csrVal[i] = val;
            csrColInd[i] = col;
            csrRowPtr[row + 1]++;
        }

        // Compute prefix sum of csrRowPtr with csrRowPtr[0] = 0
        // uint64_t stride = 1;
        // while(stride < numRows)
        // {
        //     #pragma omp parallel for
        //     for(uint64_t i = stride; i <= numRows; i += stride * 2)
        //     {
        //         csrRowPtr[i] += csrRowPtr[i - stride];
        //     }
        //     stride *= 2;
        // }
        // stride = (numRows + 1) / 2;
        // while (stride > 0)
        // {
        //     #pragma omp parallel for
        //     for(uint64_t i = stride; i <= numRows; i += stride * 2)
        //     {
        //         csrRowPtr[i + stride] += csrRowPtr[i];
        //     }
        //     stride /= 2;
        // }


        for(uint64_t i = 1; i <= numRows; i++) 
        {
            csrRowPtr[i] += csrRowPtr[i - 1];
        }

        auto end = omp_get_wtime();
        std::cout << "Read file speed: " << GET_GTEPS(end-start, EdgeNum) << " GTEPS" << std::endl;
        std::cout << "Read file time " << end-start << " s" << std::endl;
    }

    void runPageRank(double* res,COM_TYPE com_tpye = COM_TYPE::GPU, double damping = 0.85, double error_lim = EPSILON, uint64_t max_iter = 1000)
    {
        uint64_t N = VertexNum;
        double* x = (double*)malloc(N * sizeof(double));
        double init = 1.0f / N;
        for(int i = 0; i < N; i++) {
            x[i] = init;
        }
        uint64_t iterations = 0;
        double *error = (double*)malloc(N * sizeof(double));
        error[0] = 1.0f;
        double* d_Val;
        uint64_t* d_RowPtr;
        uint64_t* d_ColData;
        double* d_x;
        double* d_y;            
        double *d_error;

        // Initialize device value
        if(com_tpye == COM_TYPE::GPU)
        {
            std::cout << "init GPU" << std::endl;
            auto start = omp_get_wtime();
            cudaMalloc((void**)&d_Val, EdgeNum * sizeof(double));
            cudaMalloc((void**)&d_RowPtr, (VertexNum + 1) * sizeof(uint64_t));
            cudaMalloc((void**)&d_ColData, EdgeNum * sizeof(uint64_t));
            cudaMemcpy(d_Val, csrVal, EdgeNum * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(d_RowPtr, csrRowPtr, (VertexNum + 1) * sizeof(uint64_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_ColData, csrColInd, EdgeNum * sizeof(uint64_t), cudaMemcpyHostToDevice);
            cudaMalloc((void**)&d_x, N * sizeof(double));
            cudaMalloc((void**)&d_y, N * sizeof(double));
            cudaMemcpy(d_x, x, N * sizeof(double), cudaMemcpyHostToDevice);
            cudaMalloc((void**)&d_error, N * sizeof(double));
            auto end = omp_get_wtime();
            std::cout << "Init speed: " << GET_GTEPS(end-start, EdgeNum) << " GTEPS" << std::endl;
            std::cout << "Init time " << end-start << " s" << std::endl;
        }

        // Perform PageRank iterations
        
        std::cout << "Start PageRank" << std::endl;
        auto start = omp_get_wtime();
        while (error[0] > EPSILON && iterations < max_iter) {
            if(com_tpye == COM_TYPE::GPU)
            {
                if(isCSC == false)
                    PageRank_cuda_csr<<<(N + 255) / 256, 256>>>(d_Val, d_RowPtr, d_ColData, d_x, d_y,d_error, N, damping);
                else
                    PageRank_cuda_csc<<<(N + 255) / 256, 256>>>(d_Val, d_RowPtr, d_ColData, d_x, d_y, d_error, N, damping);
                cudaMemcpy(error, d_error,sizeof(double), cudaMemcpyDeviceToHost);
                std::swap(d_x, d_y);
            }
            else
            {
                PageRank_cpu_csr(csrVal, csrRowPtr, csrColInd, x, res, error, N);
                std::swap(x, res);
            }
            iterations++;
        }

        if(com_tpye == COM_TYPE::GPU)
        {
            cudaMemcpy(res, d_x, N * sizeof(double), cudaMemcpyDeviceToHost);
            for(auto i = 0; i < 20; i++)
            {
                std::cout << res[i] << std::endl;
            }
        }
        else
        {
            std::swap(x, res);
        }

        auto end = omp_get_wtime();
        std::cout << "PageRank speed: " << GET_GTEPS(end-start, EdgeNum * (iterations+1)) << "GTEPS" << std::endl;
        std::cout << "PageRank time " << end-start << " s" << std::endl;

        // Free memory on the device
        cudaFree(d_RowPtr);
        cudaFree(d_ColData);
        cudaFree(d_x);
        cudaFree(d_y);

        // Free memory on the host
        free(x);
    }
};


void TransitionProb(CSRMatrix &src, CSRMatrix &dst, bool isTranspose = true)
{   
    std::cout << "TransitionProb start" << std::endl;
    auto start = omp_get_wtime();
    CSRMatrix tmp;
    tmp.VertexNum = src.VertexNum;
    tmp.EdgeNum = src.EdgeNum;
    tmp.csrVal = (double*)malloc(tmp.EdgeNum * sizeof(double));
    tmp.csrRowPtr = (uint64_t*)malloc((tmp.VertexNum + 1) * sizeof(uint64_t));
    tmp.csrColInd = (uint64_t*)malloc(tmp.EdgeNum * sizeof(uint64_t));
    tmp.csrRowPtr[0] = 0;
    #pragma omp parallel for
    for(uint64_t i = 0; i < tmp.VertexNum; i++)
    {
        double sum = 0;
        for(uint64_t j = src.csrRowPtr[i]; j < src.csrRowPtr[i + 1]; j++)
        {
            sum += src.csrVal[j];
        }
        for(uint64_t j = src.csrRowPtr[i]; j < src.csrRowPtr[i + 1]; j++)
        {
            tmp.csrVal[j] = (double)src.csrVal[j] / sum;
            tmp.csrColInd[j] = src.csrColInd[j];
        }
        tmp.csrRowPtr[i + 1] = src.csrRowPtr[i + 1];
    }
    auto end = omp_get_wtime();
    std::cout << "TransitionProb(without transpose) speed: " << GET_GTEPS(end-start, tmp.EdgeNum) << " GTEPS" << std::endl;
    std::cout << "TransitionProb(without transpose) time " << end-start << " s" << std::endl;
    if(isTranspose == true)
        tmp.Transpose(dst);
    else
    {
        tmp.swap(dst);
        dst.isCSC = true;
    }
}

#endif