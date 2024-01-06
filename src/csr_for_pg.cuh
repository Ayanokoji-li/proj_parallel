#pragma once
#include <cassert>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <csr.h>

class csr_for_pg : public CSR
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
    csr_for_pg(std::string fileName) : CSR()
    {
        read_mtx(fileName);
    }
};
