#include <cuda.h>
#include <iostream>
#include <string>
#include <algorithm>
#include <omp.h>
#include "CSR.hpp"
#include "PageRank.hpp"

std::string indicator[] = {"-method", "-file", "-noTranspose"};

int main(int argc, char** argv) {
    std::string file_name = "data/web-Google.mtx";
    COM_TYPE method = COM_TYPE::GPU;
    bool isTranspose = false;
    if(argc == 1) 
    {
        std::cout << "default input file: " << file_name << std::endl;
    }
    else
    {  
        for(int i = 1; i < argc; i++)
        {
            if(argv[i] == indicator[0])
            {
                std::cout << "method: " << argv[i+1] << std::endl;
                if(std::string(argv[i+1]) == "CPU")
                {
                    method = COM_TYPE::CPU;
                }
                else if(std::string(argv[i+1]) == "GPU")
                {
                    method = COM_TYPE::GPU;
                }
                else
                {
                    std::cout << "method error" << std::endl;
                    return 0;
                }
            }
            else if(argv[i] == indicator[1])
            {
                file_name = argv[i+1];
                std::cout << "input file: " << file_name << std::endl;
            }
            else if(argv[i] == indicator[2])
            {
                isTranspose = false;
                std::cout << "no transpose" << std::endl;
            }
        }
    }

    // Initialize host value
    std::cout << "init matrix" << std::endl;
    CSRMatrix matrix(file_name);
    CSRMatrix transition{};
    TransitionProb(matrix, transition, isTranspose);
    // EFGMatrix efg();
    uint64_t N = transition.VertexNum;
    double* y = (double*)malloc(N * sizeof(double));
    transition.runPageRank(y, method, 0.85, EPSILON, 1000);
    std::cout << "run end" << std::endl;

    // free(y);
    return 0;
}