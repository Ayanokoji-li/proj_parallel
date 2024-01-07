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
    double* csrVal;
    uint64_t* csrRowPtr;
    uint64_t* csrColInd;
    uint64_t EdgeNum;
    uint64_t VertexNum;

    CSRMatrix() {};
    CSRMatrix(const std::string &fileName) {
        readFromFile(fileName);
    }

    ~CSRMatrix() {
        free(csrVal);
        free(csrRowPtr);
        free(csrColInd);
    }

    void Transpose(CSRMatrix &dst)
    {
        dst.VertexNum = VertexNum;
        dst.EdgeNum = EdgeNum;
        dst.csrVal = (double*)malloc(EdgeNum * sizeof(double));
        dst.csrRowPtr = (uint64_t*)malloc((VertexNum + 1) * sizeof(uint64_t));
        dst.csrColInd = (uint64_t*)malloc(EdgeNum * sizeof(uint64_t));

        // col_index, row_index
        COO coo(EdgeNum);
        #pragma omp parallel for
        for(uint64_t i = 0; i < VertexNum; i++)
        {
            for(uint64_t j = csrRowPtr[i]; j < csrRowPtr[i + 1]; j++)
            {
                coo[j] = std::make_tuple(csrColInd[j], i, csrVal[j]);
            }
        }
        std::stable_sort(coo.begin(), coo.end(), comp_2());
        std::stable_sort(coo.begin(), coo.end(), comp_1());

        dst.csrRowPtr[0] = 0;
        for(uint64_t i = 0; i < EdgeNum; i++)
        {
            dst.csrVal[i] = std::get<2>(coo[i]);
            dst.csrColInd[i] = std::get<1>(coo[i]);
            dst.csrRowPtr[std::get<0>(coo[i]) + 1]++;
        }
        for(uint64_t i = 1; i <= VertexNum; i++)
        {
            dst.csrRowPtr[i] += dst.csrRowPtr[i - 1];
        }
    }

    void readFromFile(const std::string &fileName)
    {
        std::ifstream file(fileName);
        std::string line;
        uint64_t numRows, numCols;

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

        // Compute row pointer array
        for (uint64_t i = 1; i <= numRows; i++) {
            csrRowPtr[i] += csrRowPtr[i - 1];
        }
    }
};


void TransitionProb(CSRMatrix &src, CSRMatrix &dst)
{
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
    tmp.Transpose(dst);
}

#endif