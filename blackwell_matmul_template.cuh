#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <iostream>
#include <vector>
#include <random>
#include "tcgen05_intrinsics.cuh" // Your provided intrinsics

using namespace blackwell::tcgen05;
using bfloat16 = __nv_bfloat16;

// --- Tuning Constants ---
// Blackwell TMEM allows very large tiles. 128x128 is a sweet spot for density.
constexpr int B_M = 128;
constexpr int B_N = 128;
constexpr int B_K = 64;   // Standard K-step for these instructions
constexpr int STAGES = 3; // Pipeline depth (3 or 4 is typical)

// Helper for TMA loading (wraps the PTX)
__device__ __forceinline__ void tma_load(void* smem_ptr, const void* tensor_map, int crds_x, int crds_y, uint64_t* mbarrier_ptr) {
    uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    uint32_t mbarrier_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(mbarrier_ptr));
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :
        : "r"(smem_int_ptr), "l"(tensor_map), "r"(crds_x), "r"(crds_y), "r"(mbarrier_int_ptr)
        : "memory"
    );
}

// QUESTION: explain this to me
__device__ __forceinline__ void mbarrier_init(uint64_t* barrier, uint32_t expected_bytes) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(barrier));
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(smem_addr), "r"(expected_bytes) : "memory");
}

// QUESTION: explain this to me
__device__ __forceinline__ void mbarrier_wait(uint64_t* barrier, int phase) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(barrier));
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n\t"
        "@!p nanosleep.u32 20;\n\t" 
        "@!p bra LAB_WAIT;\n\t"
        "}"
        : : "r"(smem_addr), "r"(phase)
    );
}

__global__ void __launch_bounds__(128) blackwell_matmul_kernel(
    const __grid_constant__ CUtensorMap tensorMapA,
    const __grid_constant__ CUtensorMap tensorMapB,
    bfloat16* C,
    int M, int N, int K
) {
    // ----------------------------------------------------------------
    // 1. Setup & Address Calculation
    // ----------------------------------------------------------------
    int tid = threadIdx.x;
    int bx  = blockIdx.x; // Tile N
    int by  = blockIdx.y; // Tile M
    
    // Shared Memory Layout
    // We need buffers for A and B for each pipeline stage.
    extern __shared__ __align__(1024) uint8_t smem[];
    bfloat16* smem_A = reinterpret_cast<bfloat16*>(smem);
    
    // Offset smem_B to start after all stages of A
    bfloat16* smem_B = smem_A + (STAGES * B_M * B_K);

    // Barriers live at the end
    uint64_t* barrier = reinterpret_cast<uint64_t*>(smem_B + (STAGES * B_K * B_N));
    uint64_t* mma_barrier = &barrier[STAGES];

    // ----------------------------------------------------------------
    // 2. Initialization (One thread per block usually does this)
    // ----------------------------------------------------------------
    uint32_t transaction_bytes = (B_M * B_K * sizeof(bfloat16)) + (B_K * B_N * sizeof(bfloat16));
    if (tid < STAGES) {
        mbarrier_init(&barrier[tid], transaction_bytes);
    }
    if (tid == STAGES) {
        mbarrier_init(mma_barrier, 32); // Expect 32 commits (1 per thread in warp 0)
    }
    __syncthreads();

    // ----------------------------------------------------------------
    // 3. TMEM Allocation (Warp 0 only)
    // ----------------------------------------------------------------
    if (tid < 32) {
        // Blackwell specific: Allocate Tensor Memory columns.
        // 32 columns is typical for a 128x128 tile in this mode.
        alloc(32, smem);
    }
    // Note: alloc is technically warp-synchronous, but we often sync block here
    // to ensure allocation is done before anyone tries to issue math.
    __syncthreads();

    // ----------------------------------------------------------------
    // 4. Pipeline Prologue (Fill the pipeline)
    // ----------------------------------------------------------------
    // We want to issue loads for the first (STAGES - 1) tiles.
    
    // Loop s from 0 to STAGES - 2
    for (int s = 0; s < STAGES - 1; ++s) {
        if (tid == 0) {
            // A: Row Major (M, K) -> (Row, Col) = (by*B_M, s*B_K)
            // tma_load map coords: (Col/K, Row/M) for RowMajor A
            tma_load(
                smem_A + s * (B_M * B_K), 
                &tensorMapA, 
                s * B_K,     // Col/K
                by * B_M,    // Row/M
                &barrier[s]
            );

            // B: Col Major (K, N) -> (Row, Col) = (s*B_K, bx*B_N)
            // tma_load map coords: (Row/K, Col/N) for ColMajor B
            tma_load(
                smem_B + s * (B_K * B_N), 
                &tensorMapB, 
                s * B_K,     // Row/K
                bx * B_N,    // Col/N
                &barrier[s]
            );
        }
    }

    // ----------------------------------------------------------------
    // 5. Main Loop
    // ----------------------------------------------------------------
    int num_tiles_k = K / B_K;
    int pipeline_stage = 0; // Circular buffer index (0 to STAGES-1)
    
    // TMEM address for our accumulator (usually 0 if we only have 1 tile)
    uint32_t tmem_accum = 0; 

    for (int k = 0; k < num_tiles_k; ++k) {
        
        // A. Issue Next Load (Producer)
        // Check if we have remaining tiles to load (k + STAGES - 1)
        if (k + STAGES - 1 < num_tiles_k) {
            if (tid == 0) {
                int next_k_tile = k + STAGES - 1;
                int next_stage = (pipeline_stage + STAGES - 1) % STAGES;

                // Load A (Row Major: Col, Row)
                tma_load(
                    smem_A + next_stage * (B_M * B_K), 
                    &tensorMapA, 
                    next_k_tile * B_K, 
                    by * B_M, 
                    &barrier[next_stage]
                );

                // Load B (Col Major: Row, Col)
                tma_load(
                    smem_B + next_stage * (B_K * B_N), 
                    &tensorMapB, 
                    next_k_tile * B_K, 
                    bx * B_N, 
                    &barrier[next_stage]
                );
            }
        }

        // B. Wait for Data (Consumer)
        // Phase parity: flips every STAGES iterations
        mbarrier_wait(&barrier[pipeline_stage], (k / STAGES) % 2);

        // C. Issue Math (Consumer)
        // On Blackwell, only one warp (or even one thread) needs to issue the MMA.
        // It broadcasts the command to the Tensor Cores.
        if (tid < 32) { // Leader Warp
            // 1. Create Descriptors
            uint64_t desc_a = make_smem_desc(smem_A + pipeline_stage * (B_M * B_K), SWIZZLE_128B);
            uint64_t desc_b = make_smem_desc(smem_B + pipeline_stage * (B_K * B_N), SWIZZLE_128B);

            // 2. Issue MMA
            // Param 'accumulate': (k > 0) ? true : false
            mma_128x128x64_bf16(tmem_accum, desc_a, desc_b, (k > 0));
            commit_mbarrier(mma_barrier);

        }

        // D. Pipeline Management
        // Advance stage index
        pipeline_stage = (pipeline_stage + 1) % STAGES;
    }

    // ----------------------------------------------------------------
    // 6. Epilogue
    // ----------------------------------------------------------------
    
    // Wait for all MMA operations to complete
    if (tid < 32) {
        wait_mbarrier(mma_barrier, 0);
    }

    __syncthreads();

    // Load from TMEM -> Registers -> Global Memory
    // Each thread loads a fragment (4 floats = 128 bits)
    // QUESTION: how many threads do we actually need to load from TMEM? are we using the correct number?
    float frag[4];
    
    // Calculate offset in TMEM (This is tricky!)
    // Standard mapping: tid * 16 bytes (for 128b load)
    uint32_t tmem_offset = tid * 16; // Simplified linear mapping

    ld_16x128b_x4(frag, tmem_accum + tmem_offset);

    // Store to C
    // Simplified store for verification
    int row = by * B_M + (tid / 32); 
    int col = bx * B_N + (tid % 32) * 4; 
    
    if (row < M && col + 3 < N) {
        // Just storing first element for now
        C[row * N + col] = __float2bfloat16(frag[0]);
    }

    // Cleanup
    if (tid < 32) {
        relinquish_alloc_permit();
        dealloc(0, 32);
    }
}

// --- Host Helpers ---

inline void checkCuda(cudaError_t result, const char* func, const char* file, int line) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Error at " << file << ":" << line << " - " << func << ": " 
                  << cudaGetErrorString(result) << std::endl;
        exit(1);
    }
}
#define CHECK_CUDA(val) checkCuda((val), #val, __FILE__, __LINE__)

inline void checkCu(CUresult result, const char* func, const char* file, int line) {
    if (result != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(result, &errStr);
        std::cerr << "CUDA Driver Error at " << file << ":" << line << " - " << func << ": " 
                  << errStr << std::endl;
        exit(1);
    }
}
#define CHECK_CU(val) checkCu((val), #val, __FILE__, __LINE__)

// Helper to initialize random BF16 data
inline void init_bf16(bfloat16* ptr, size_t count, float val = -1.0f) {
    std::vector<bfloat16> data(count);
    for (size_t i = 0; i < count; ++i) {
        float v = (val == -1.0f) ? (static_cast<float>(rand()) / RAND_MAX) : val;
        data[i] = __float2bfloat16(v);
    }
    CHECK_CUDA(cudaMemcpy(ptr, data.data(), count * sizeof(bfloat16), cudaMemcpyHostToDevice));
}

// --- TMA Setup Helper ---
// This creates the "Texture Object" equivalent for Tensor Memory

enum class TensorLayout {
    ROW_MAJOR,
    COL_MAJOR
};

inline void create_tma_descriptor(
    CUtensorMap* tma_map, 
    void* global_address, 
    int dim_row, int dim_col, // Global Matrix Dimensions
    int tile_row, int tile_col,  // Tile Dimensions (e.g., 128x64)
    TensorLayout layout = TensorLayout::ROW_MAJOR
) {
    // 1. Define the global tensor size
    cuuint64_t globalDims[2];
    cuuint64_t globalStrides[1];
    cuuint32_t boxDims[2];
    
    if (layout == TensorLayout::ROW_MAJOR) {
        // Row Major: Dim 0 is Col (fast), Dim 1 is Row
        globalDims[0] = (cuuint64_t)dim_col; // Width
        globalDims[1] = (cuuint64_t)dim_row; // Height
        
        // Stride for traversing rows (Dim 1)
        globalStrides[0] = (cuuint64_t)dim_col * sizeof(bfloat16); 
        
        boxDims[0] = (cuuint32_t)tile_col;
        boxDims[1] = (cuuint32_t)tile_row;
    } else {
        // Column Major: Dim 0 is Row (fast), Dim 1 is Col
        globalDims[0] = (cuuint64_t)dim_row; // Height
        globalDims[1] = (cuuint64_t)dim_col; // Width
        
        // Stride for traversing columns (Dim 1)
        globalStrides[0] = (cuuint64_t)dim_row * sizeof(bfloat16);
        
        boxDims[0] = (cuuint32_t)tile_row;
        boxDims[1] = (cuuint32_t)tile_col;
    }
    
    // 4. Element stride (1 for standard dense)
    cuuint32_t elementStrides[2] = { 1, 1 }; 

    // 5. Create the Map
    // Note: Rank 2 means 2D tensor.
    CHECK_CU(cuTensorMapEncodeTiled(
        tma_map,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, // Data Type
        2,                                   // Tensor Rank (2D)
        global_address,                      // Global Memory Pointer
        globalDims,
        globalStrides + 0,                   // Strides (rank - 1)
        boxDims,
        elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,   // Interleaving (used for some specific layouts)
        CU_TENSOR_MAP_SWIZZLE_128B,      // Swizzle bank conflicts (Important for TMA!)
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, // L2 Cache policy
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE   // OOB behavior
    ));
}

int main() {
    // 1. Setup Device
    int dev = 0;
    cudaSetDevice(dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    std::cout << "Running on: " << prop.name << " (SM " << prop.major << "." << prop.minor << ")" << std::endl;

    if (prop.major < 10) {
        std::cout << "WARNING: This code requires SM100 (Blackwell) architecture." << std::endl;
    }

    // 2. Problem Size
    // Must be multiples of tile sizes for this simplified kernel
    int M = 4096;
    int N = 4096;
    int K = 4096;

    std::cout << "Matrix Size: " << M << "x" << N << "x" << K << std::endl;

    size_t bytes_A = M * K * sizeof(bfloat16);
    size_t bytes_B = K * N * sizeof(bfloat16);
    size_t bytes_C = M * N * sizeof(bfloat16);

    // 3. Allocate Memory
    bfloat16 *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, bytes_A));
    CHECK_CUDA(cudaMalloc(&d_B, bytes_B));
    CHECK_CUDA(cudaMalloc(&d_C, bytes_C));

    // 4. Initialize Data
    init_bf16(d_A, M * K);
    init_bf16(d_B, K * N);
    CHECK_CUDA(cudaMemset(d_C, 0, bytes_C));

    // 5. Setup TMA Descriptors
    // The kernel defines B_M=128, B_N=128, B_K=64
    // Map A: (M x K) loading tiles of (B_M x B_K) -> (128 x 64)
    CUtensorMap tma_map_A;
    create_tma_descriptor(&tma_map_A, d_A, M, K, B_M, B_K, TensorLayout::ROW_MAJOR);

    // Map B: (K x N) loading tiles of (B_K x B_N) -> (64 x 128)
    // Note: B is Column Major (K rows, N cols). 
    // We load (K_step x N_step) tiles.
    CUtensorMap tma_map_B;
    create_tma_descriptor(&tma_map_B, d_B, K, N, B_K, B_N, TensorLayout::COL_MAJOR);

    // 6. Launch Config
    // Grid: Covers the output matrix C (M x N) in tiles of 128x128
    dim3 grid(N / B_N, M / B_M, 1);
    
    // Block: 128 threads (4 Warps). 
    dim3 block(128, 1, 1);

    // Shared Memory Calculation
    // We need: STAGES * (TileA + TileB) + Barriers
    size_t tile_A_bytes = B_M * B_K * sizeof(bfloat16);
    size_t tile_B_bytes = B_K * B_N * sizeof(bfloat16);
    size_t barrier_bytes = (STAGES + 1) * sizeof(uint64_t); // 8 bytes per barrier
    
    size_t smem_size = STAGES * (tile_A_bytes + tile_B_bytes) + barrier_bytes;
    
    // Ensure we can allocate this much Shared Mem (Blackwell has plenty)
    CHECK_CUDA(cudaFuncSetAttribute(
        blackwell_matmul_kernel, 
        cudaFuncAttributeMaxDynamicSharedMemorySize, 
        smem_size
    ));

    std::cout << "Launching Kernel..." << std::endl;
    std::cout << "Grid: " << grid.x << "x" << grid.y << " Block: " << block.x << std::endl;
    std::cout << "Smem: " << smem_size / 1024 << " KB" << std::endl;

    // 7. Launch
    blackwell_matmul_kernel<<<grid, block, smem_size>>>(
        tma_map_A,
        tma_map_B,
        d_C,
        M, N, K
    );
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    std::cout << "Kernel Completed Successfully." << std::endl;

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
