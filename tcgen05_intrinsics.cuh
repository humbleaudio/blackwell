#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace blackwell {
namespace tcgen05 {

// --------------------------------------------------------------------------
// Tensor Memory (TMEM) Management
// --------------------------------------------------------------------------

/**
 * @brief Allocates TMEM columns for the CTA group.
 * 
 * @param ncols Number of columns to allocate.
 * @param smem_ptr Pointer to shared memory to store the allocated TMEM address.
 */
__device__ __forceinline__ void alloc(uint32_t ncols, void* smem_ptr) {
    uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        :
        : "r"(smem_int_ptr), "r"(ncols)
        : "memory"
    );
}

/**
 * @brief Deallocates TMEM columns.
 * 
 * @param tmem_addr Address of TMEM to deallocate.
 * @param ncols Number of columns to deallocate.
 */
__device__ __forceinline__ void dealloc(uint32_t tmem_addr, uint32_t ncols) {
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
        :
        : "r"(tmem_addr), "r"(ncols)
        : "memory"
    );
}

/**
 * @brief Relinquishes the allocation permit (releases ownership).
 *        Typically called before dealloc in the cleanup phase.
 */
__device__ __forceinline__ void relinquish_alloc_permit() {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
}


// --------------------------------------------------------------------------
// Synchronization & Commit
// --------------------------------------------------------------------------

/**
 * @brief Commits issued TCGEN05 operations to the mbarrier.
 *        Ensures operations are tracked for completion.
 * @param mbarrier Pointer to the mbarrier in shared memory.
 */
__device__ __forceinline__ void commit_mbarrier(uint64_t* mbarrier) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbarrier));
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
        : : "r"(smem_addr) : "memory"
    );
} 

/**
 * @brief Waits for the mbarrier to complete.
 * 
 * @param barrier Pointer to the mbarrier in shared memory.
 * @param phase Phase of the mbarrier.
 */
__device__ __forceinline__ void wait_mbarrier(uint64_t* barrier, int phase) {
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
/**
 * @brief Deprecated: Use commit_mbarrier + mbarrier_wait instead.
 *        This function previously attempted to use tcgen05.wait which is invalid for MMA.
 */
// __device__ __forceinline__ void wait_mma() { ... }



// --------------------------------------------------------------------------
// Matrix Multiply-Accumulate (MMA)
// --------------------------------------------------------------------------

/**
 * @brief Performs a 128x128x64 BF16 Matrix Multiplication Accumulating into TMEM.
 * 
 * TMEM[tmem_addr] = A * B + (accumulate ? TMEM[tmem_addr] : 0)
 * 
 * @param tmem_addr Address offset in TMEM for the accumulator tile.
 * @param desc_a    Shared memory descriptor for Matrix A (64-bit).
 * @param desc_b    Shared memory descriptor for Matrix B (64-bit).
 * @param accumulate If true, adds to existing TMEM content. If false, overwrites (effectively C = A*B).
 *                   (Note: The '0' predicate in asm usually controls this, 
 *                    0 = overwrite/disable input D? Check specific ISA semantics.
 *                    In the reference, 0 was passed as the predicate. 
 *                    Often '0' means false -> C = A*B. '1' -> C += A*B.)
 */
__device__ __forceinline__ void mma_128x128x64_bf16(uint32_t tmem_addr, uint64_t desc_a, uint64_t desc_b, bool accumulate = false) {
    // idesc: Instruction descriptor (controls swizzle, sparsity, scale). 
    // 0 is the default for dense operations.
    uint32_t idesc = 0; 
    uint32_t mask[4] = {0, 0, 0, 0}; // Dummy mask for unused D registers

    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.u32 p, %4, 0;\n\t" // p = (accumulate != 0)
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, {%5, %6, %7, %8}, p; \n\t"
        "}\n"
        :
        : "r"(tmem_addr), "l"(desc_a), "l"(desc_b), "r"(idesc), 
          "r"((int)accumulate), 
          "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3])
        : "memory"
    );
}

// --------------------------------------------------------------------------
// Data Movement (TMEM Load/Store/Copy)
// --------------------------------------------------------------------------

/**
 * @brief Copies data from Shared Memory to TMEM.
 *        Useful for initializing TMEM or loading bias/accumulators.
 * 
 * @param tmem_addr Address offset in TMEM.
 * @param smem_desc Shared memory descriptor (64-bit) for the source data.
 */
__device__ __forceinline__ void cp_128x256b(uint32_t tmem_addr, uint64_t smem_desc) {
    asm volatile(
        "tcgen05.cp.cta_group::1.128x256b [%0], %1;"
        :
        : "r"(tmem_addr), "l"(smem_desc)
        : "memory"
    );
}

/**
 * @brief Loads 4 floats (128 bits) from TMEM into registers.
 *        Used in the epilogue to retrieve results.
 * 
 * @param dst Pointer to float[4] array in registers.
 * @param tmem_addr Byte offset in TMEM to load from.
 */
__device__ __forceinline__ void ld_16x128b_x4(float* dst, uint32_t tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.16x128b.x2.b32 "
        "{%0, %1, %2, %3}, [%4];"
        : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3])
        : "r"(tmem_addr)
        : "memory"
    );
}


// --------------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------------

// Swizzle Modes for SMEM Descriptor (Bits 61-63)
enum SwizzleMode {
    SWIZZLE_NONE = 0,
    SWIZZLE_128B = 2,
    SWIZZLE_64B  = 4,
    SWIZZLE_32B  = 6
};

/**
 * @brief Creates a 64-bit Shared Memory Descriptor for TMA/TCGEN05 operations.
 *        Matches CUTLASS/CuTe SM100 descriptor format.
 * 
 * @param ptr Pointer to shared memory buffer.
 * @param swizzle_mode Swizzle mode (default SWIZZLE_128B). 
 *                     Use values from SwizzleMode enum.
 * @return uint64_t The descriptor.
 */
__device__ __forceinline__ uint64_t make_smem_desc(const void* ptr, int swizzle_mode = SWIZZLE_128B) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t desc = 0;
    
    // 1. Start Address (Bits 0-13): Address >> 4
    desc |= (uint64_t)(addr >> 4); 

    // 2. Swizzle Mode (Bits 61-63)
    desc |= ((uint64_t)swizzle_mode << 61); 

    // 3. Version (Bits 46-47): Set to 1 for Blackwell
    // Matches CuTe SmemDescriptor.version_ = 1
    desc |= ((uint64_t)1 << 46);

    return desc;
}

} // namespace tcgen05
} // namespace blackwell
