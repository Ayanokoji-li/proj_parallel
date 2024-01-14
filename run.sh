#! /usr/bin/bash

nvcc -O3 -Xcompiler -fopenmp main.cu -o main

./main

# change method
# -method {CPU/GPU}

# default with transpose
# no transpose
# -noTranspose

# change input file
# -file {path}