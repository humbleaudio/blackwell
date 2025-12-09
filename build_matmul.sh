#!/bin/bash

# Configuration
# Explicitly add CUDA bin to PATH if needed, or use system default
# export PATH=/usr/local/cuda-13.0/bin:$PATH
NVCC=nvcc

SOURCE="blackwell_matmul_template.cuh"
OUTPUT="blackwell_matmul"

# Compilation
# We target compute_100a to ensure PTX version supports tcgen05 features,
# and code=sm_100a for the actual binary.

echo "Compiling $SOURCE for SM100A..."

$NVCC -x cu $SOURCE -o $OUTPUT \
    -gencode arch=compute_100a,code=sm_100a \
    -lcublas \
    -lcuda \
    -std=c++17 \
    -O3 \
    --ptxas-options=-v \
    -keep

if [ $? -eq 0 ]; then
    echo "Build successful! Executable: ./$OUTPUT"
else
    echo "Build failed!"
    exit 1
fi
