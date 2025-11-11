#!/bin/bash

cd /home/will/git/fast.cu && cat > /tmp/check_smem.cu << 'EOF'
#include <cuda_runtime.h>
#include <stdio.h>
int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device %d: %s\n", i, prop.name);
        printf("  Shared Memory per Block: %zu bytes\n", prop.sharedMemPerBlock);
        printf("  Shared Memory per SM: %zu bytes\n", prop.sharedMemPerMultiprocessor);
        printf("  Shared Memory per Block Optin: %zu bytes\n", prop.sharedMemPerBlockOptin);
        printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
        printf("  Number of SMs: %d\n", prop.multiProcessorCount);
        printf("  Registers per Block: %d\n", prop.regsPerBlock);
        printf("  Registers per SM: %d\n", prop.regsPerMultiprocessor);
        printf("  Warp Size: %d\n", prop.warpSize);
    }
    return 0;
}
EOF
nvcc /tmp/check_smem.cu -o /tmp/check_smem && /tmp/check_smem