#include <iostream>
#include <fstream>
#include <vector>

// Function to read Matrix Market format matrix
void readMatrixMarket(const std::string& filePath, std::vector<int>& rowIndices, std::vector<int>& columnIndices, std::vector<double>& values, int& numRows, int& numColumns) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cout << "Failed to open file: " << filePath << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line[0] == '%') {
            continue; // Skip comment lines
        }

        std::istringstream iss(line);
        int row, col;
        double value;
        iss >> row >> col >> value;

        rowIndices.push_back(row);
        columnIndices[col + 1]++;
        values.push_back(value);

        numRows = std::max(numRows, row + 1);
        numColumns = std::max(numColumns, col + 1);
    }

    file.close();
}

// Function to multiply CSC matrix with a vector
std::vector<double> multiplyCSCWithVector(const std::vector<int>& rowIndices, const std::vector<int>& columnIndices, const std::vector<double>& values, const std::vector<double>& vector) {
    int numRows = rowIndices.size() - 1;
    int numColumns = columnIndices.size();

    std::vector<double> result(numRows, 0.0);

    for (int col = 0; col < numColumns; col++) {
        int start = columnIndices[col];
        int end = columnIndices[col + 1];

        for (int i = start; i < end; i++) {
            int row = rowIndices[i];
            double value = values[i];

            result[row] += value * vector[col];
        }
    }

    return result;
}

int main() {
    std::string filePath = "./data/web-Google.mtx";

    std::vector<int> rowIndices;
    std::vector<int> columnIndices;
    std::vector<double> values;
    int numRows = 0;
    int numColumns = 0;

    readMatrixMarket(filePath, rowIndices, columnIndices, values, numRows, numColumns);

    std::vector<double> vector(numColumns, 1.0);

    std::vector<double> result = multiplyCSCWithVector(rowIndices, columnIndices, values, vector);

    // Print the result
    for (int i = 0; i < numRows; i++) {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
