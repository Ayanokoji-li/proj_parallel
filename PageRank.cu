#include <cuda.h>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>

// suppose weight is integer
struct CSRMatrix {
    std::vector<unsigned long long> csrVal;
    std::vector<unsigned long long> csrRowPtr, csrColInd;
    unsigned long long MaxVal;
    unsigned long long numNonzeros;

    CSRMatrix() {};
    CSRMatrix(const std::string &fileName) {
        readFromFile(fileName);
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
        s >> numRows >> numCols >> numNonzeros;

        csrRowPtr.resize(numRows + 1, 0);
        csrVal.resize(numNonzeros);
        csrColInd.resize(numNonzeros);

        unsigned long long row, col;
        unsigned long long val;
        for (int i = 0; i < numNonzeros; i++) {
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


class EFGMatrix {
public:
    std::vector<unsigned long long> vlist; // Vertex list
    std::vector<int> num_lower_bits; // Number of lower bits used for EF encoding
    std::vector<int> offsets; // Offsets into the data array
    std::vector<unsigned long long> data; // EF encoded data

    // Constructor from a CSR matrix
    EFGMatrix(const std::string &fileName) {
        initFromCSR(fileName);
    }

private:
    // Initialize the EFGMatrix from a CSR matrix
    void initFromCSR(const std::string &fileName) {
        CSRMatrix csr(fileName);

        // Initialize vlist array
        vlist = csr.csrRowPtr;

        // Initialize num_lower_bits, offsets and data arrays
        num_lower_bits.resize(csr.csrRowPtr.size() - 1);
        offsets.resize(csr.csrRowPtr.size() - 1);
        data.clear();

        for (int i = 0; i < csr.csrRowPtr.size() - 1; i++) {
            // Get the neighbor list for vertex i
            std::vector<int> neighbors(csr.csrColInd.begin() + csr.csrRowPtr[i], csr.csrColInd.begin() + csr.csrRowPtr[i + 1]);

            // Compute the number of lower bits for EF encoding
            num_lower_bits[i] = computeNumLowerBits(neighbors);

            // Compute the offset into the data array
            offsets[i] = data.size();

            // Encode the neighbor list with EF and append it to the data array
            std::vector<unsigned long long> encodedNeighbors = encodeWithEF(neighbors, num_lower_bits[i]);
            data.insert(data.end(), encodedNeighbors.begin(), encodedNeighbors.end());
        }
    }

    // Compute the number of lower bits for EF encoding
    int computeNumLowerBits(const std::vector<int> &neighbors) {
        // Compute the maximum value in the neighbor list
        int max_value = *std::max_element(neighbors.begin(), neighbors.end());

        // Compute the number of lower bits
        int num_lower_bits = max(0, (int)floor(log2(max_value / neighbors.size())));

        return num_lower_bits;
    }

    // Encode a neighbor list with EF
    std::vector<unsigned long long> encodeWithEF(const std::vector<int> &neighbors, int num_lower_bits) {
        std::vector<unsigned long long> encodedNeighbors;

        // Encode each neighbor
        for (int neighbor : neighbors) {
            // Split the neighbor into lower and upper bits
            unsigned long long lower_bits = neighbor & ((1 << num_lower_bits) - 1);
            unsigned long long upper_bits = neighbor >> num_lower_bits;

            // Encode the lower bits
            encodedNeighbors.push_back(lower_bits);

            // Encode the upper bits as unary coded gaps
            if (upper_bits > 0) {
                encodedNeighbors.push_back((1 << upper_bits) - 1);
            }
        }

        return encodedNeighbors;
    }
};