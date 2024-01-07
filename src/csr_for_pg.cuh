#pragma once
#include <cassert>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "csr.h"

class CSR_for_pg : public CSR
{
private:
    std::vector<double> csrVal;
    
    int read_mtx(std::string fileName)
    {
        std::ifstream file(fileName);
        std::string line;
        uint64_t numRows, numCols;

        // Skip header
        do {
            std::getline(file, line);
        } while (line[0] == '%');

        std::stringstream s(line);
        s >> numRows >> numCols >> num_edges_;
        num_vertices_ = numRows;
        csrVal.resize(num_edges_);
        elist_.resize(num_edges_);
        vlist_.resize(num_vertices_ + 1, 0);

        unsigned long long row, col;
        unsigned long long val;
        for (unsigned long long i = 0; i < num_vertices_; i++) {
            file >> row >> col >> val;
            row--;  // Convert to 0-based index
            col--;
            csrVal[i] = val;
            elist_[i] = col;
            vlist_[row + 1]++;
        }

        // Compute row pointer array
        for (unsigned long long i = 1; i <= numRows; i++) {
            vlist_[i] += vlist_[i - 1];
        }
    }

public:   
    CSR_for_pg(std::string fileName) : CSR()
    {
        read_mtx(fileName);
    }

    void transpose()
    {
        std::vector<uint64_t> vlist_t(num_vertices_ + 1, 0);
        std::vector<uint64_t> elist_t(num_edges_);
        std::vector<double> csrVal_t(num_edges_);

        for (unsigned long long i = 0; i < num_vertices_; i++) {
            for (unsigned long long j = vlist_[i]; j < vlist_[i + 1]; j++) {
                vlist_t[elist_[j] + 1]++;
            }
        }

        for (unsigned long long i = 1; i <= num_vertices_; i++) {
            vlist_t[i] += vlist_t[i - 1];
        }

        for (unsigned long long i = 0; i < num_vertices_; i++) {
            for (unsigned long long j = vlist_[i]; j < vlist_[i + 1]; j++) {
                unsigned long long k = vlist_t[elist_[j]]++;
                elist_t[k] = i;
                csrVal_t[k] = csrVal[j];
            }
        }

        vlist_ = vlist_t;
        elist_ = elist_t;
        csrVal = csrVal_t;
    }
};
