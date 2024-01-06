#ifndef CSR_HPP
#define CSR_HPP

#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <omp.h>

template <typename T>
using COO = std::vector<std::tuple<unsigned long long, unsigned long long, T>>;

template <typename T>
struct comp_1 {
    bool operator()(const std::tuple<unsigned long long, unsigned long long, T> &a, const std::tuple<unsigned long long, unsigned long long, T> &b) {
        return std::get<0>(a) < std::get<0>(b);
    }
};

template <typename T>
struct comp_2 {
    bool operator()(const std::tuple<unsigned long long, unsigned long long, T> &a, const std::tuple<unsigned long long, unsigned long long, T> &b) {
        return std::get<1>(a) < std::get<1>(b);
    }
};


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

        // col_index, row_index
        COO<T> coo(EdgeNum);
        #pragma omp parallel for
        for(unsigned long long i = 0; i < VertexNum; i++)
        {
            for(unsigned long long j = csrRowPtr[i]; j < csrRowPtr[i + 1]; j++)
            {
                coo[j] = std::make_tuple(csrColInd[j], i, csrVal[j]);
            }
        }
        std::stable_sort(coo.begin(), coo.end(), comp_2<T>());
        std::stable_sort(coo.begin(), coo.end(), comp_1<T>());

        dst.csrRowPtr[0] = 0;
        for(unsigned long long i = 0; i < EdgeNum; i++)
        {
            dst.csrVal[i] = std::get<2>(coo[i]);
            dst.csrColInd[i] = std::get<1>(coo[i]);
            dst.csrRowPtr[std::get<0>(coo[i]) + 1]++;
        }
        for(unsigned long long i = 1; i <= VertexNum; i++)
        {
            dst.csrRowPtr[i] += dst.csrRowPtr[i - 1];
        }
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
        csrRowPtr = (unsigned long long*)malloc((numRows + 1) * sizeof(unsigned long long));
        csrVal = (T*)malloc(EdgeNum * sizeof(T));
        csrColInd = (unsigned long long*)malloc(EdgeNum * sizeof(unsigned long long));

        unsigned long long row, col;
        unsigned long long val;
        for (unsigned long long i = 0; i < EdgeNum; i++) {
            file >> row >> col >> val;
            row--;  // Convert to 0-based index
            col--;
            csrVal[i] = val;
            csrColInd[i] = col;
            csrRowPtr[row + 1]++;
        }

        // Compute row pointer array
        for (unsigned long long i = 1; i <= numRows; i++) {
            csrRowPtr[i] += csrRowPtr[i - 1];
        }
    }

    void TransitionProb(CSRMatrix<double> &dst)
    {
        TransitionProb(*this, dst);
    }
};

template <typename T>
void TransitionProb(CSRMatrix<T> &src, CSRMatrix<double> &dst)
{
    dst.VertexNum = src.VertexNum;
    dst.EdgeNum = src.EdgeNum;
    dst.csrVal = (double*)malloc(dst.EdgeNum * sizeof(double));
    dst.csrRowPtr = (unsigned long long*)malloc((dst.VertexNum + 1) * sizeof(unsigned long long));
    dst.csrColInd = (unsigned long long*)malloc(dst.EdgeNum * sizeof(unsigned long long));
    dst.csrRowPtr[0] = 0;
    for(unsigned long long i = 0; i < dst.VertexNum; i++)
    {
        T sum = 0;
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

#endif