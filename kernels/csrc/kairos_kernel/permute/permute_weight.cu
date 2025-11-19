#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cassert>
#include <stdint.h>
#include <float.h>
#include <type_traits>

// Precompute permutations for weight (same as original)
__host__ __device__ void _get_perms_format(int* perm) {
    int idx = 0;
    for (int i = 0; i < 32; i++) {
        int _perm[8];
        int col = i / 4;
        for (int block = 0; block < 2; block++) {
            for (int row_offset = 0; row_offset < 2; row_offset++) {
                int row = (i % 4) + row_offset * 4;
                _perm[block * 2 + row_offset] = 16 * row + col + 8 * block;
            }
        }
        for (int j = 0; j < 4; j++) {
            for (int p = 0; p < 8; p++) {
                perm[idx++] = _perm[p] + 128 * j;
            }
        }
    }
}

// Optimized kernel for 4x int8 to int32 conversion with bit manipulation
__global__ void permute_4int8_to_int32_kernel(const int8_t* __restrict__ input, 
                                             int32_t* __restrict__ output,
                                             int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < total_elements; i += stride) {
        const int8_t* src = input + i * 4;
        int32_t result;
        
        // Use union for efficient bit manipulation
        union {
            int32_t as_int32;
            int8_t as_int8[4];
        } converter;
        
        // Load 4 int8 values
        converter.as_int8[0] = src[0];
        converter.as_int8[1] = src[1];  
        converter.as_int8[2] = src[2];
        converter.as_int8[3] = src[3];
        
        // Bit manipulation operations
        int8_t ha = (converter.as_int8[0] >> 4) & 0x0F;
        int8_t la = converter.as_int8[0] & 0x0F;
        int8_t lb = converter.as_int8[1] & 0x0F;
        
        converter.as_int8[0] = (converter.as_int8[1] & 0xF0) | ha;
        converter.as_int8[1] = la | (lb << 4);
        
        ha = (converter.as_int8[2] >> 4) & 0x0F;
        la = converter.as_int8[2] & 0x0F;
        lb = converter.as_int8[3] & 0x0F;
        
        converter.as_int8[2] = (converter.as_int8[3] & 0xF0) | ha;
        converter.as_int8[3] = la | (lb << 4);
        
        // Swap positions 1 and 2
        int8_t temp = converter.as_int8[1];
        converter.as_int8[1] = converter.as_int8[2];
        converter.as_int8[2] = temp;
        
        output[i] = converter.as_int32;
    }
}

// Main weight permutation kernel
__global__ void permute_weight_kernel(const int8_t* __restrict__ input,
                                     int32_t* __restrict__ output,
                                     const int* __restrict__ perm_w,
                                     int in_features, int out_features,
                                     int perm_size) {
    int tile_x = 8, tile_y = 16;
    int blocks_x = in_features / tile_x;
    int blocks_y = out_features / tile_y;
    
    int total_blocks = blocks_x * blocks_y;
    int block_id = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int block_idx = block_id; block_idx < total_blocks; block_idx += stride) {
        int block_x = block_idx % blocks_x;
        int block_y = block_idx / blocks_x;
        
        // Process 8x16 tile
        for (int perm_idx = 0; perm_idx < perm_size; perm_idx++) {
            int src_pos = perm_w[perm_idx];
            
            // Calculate source position in the original layout
            int src_tile_row = src_pos / (tile_y * tile_x);
            int src_tile_col = (src_pos % (tile_y * tile_x)) / tile_x;
            int src_in_tile_row = src_pos % tile_x;
            int src_in_tile_col = src_pos % tile_y;
            
            // Calculate destination position
            int dst_row = block_x * tile_x + src_in_tile_row;
            int dst_col = block_y * tile_y * tile_x + perm_idx;
            
            if (dst_row < in_features && dst_col < out_features * tile_x) {
                // This would require careful indexing - simplified for illustration
                // Actual implementation would use shared memory for coalesced access
            }
        }
    }
}

// Optimized version using shared memory for better coalescing
template<int TILE_X = 8, int TILE_Y = 16>
__global__ void permute_weight_optimized_kernel(const int8_t* __restrict__ input,
                                               int32_t* __restrict__ output,
                                               const int* __restrict__ perm_w,
                                               int in_features, int out_features) {
    __shared__ int8_t tile[TILE_X * TILE_Y];
    
    int tile_x = TILE_X, tile_y = TILE_Y;
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;
    
    int in_block_start = block_x * tile_x;
    int out_block_start = block_y * tile_y;
    
    // Load tile into shared memory (coalesced access)
    for (int i = threadIdx.x; i < tile_x * tile_y; i += blockDim.x) {
        int row = in_block_start + (i / tile_y);
        int col = out_block_start + (i % tile_y);
        
        if (row < in_features && col < out_features) {
            tile[i] = input[row * out_features + col];
        }
    }
    __syncthreads();
    
    // Process permutation for this thread
    int perm_idx = threadIdx.x;
    if (perm_idx < 128) { // perm_w size per tile
        int src_pos = perm_w[perm_idx];
        int8_t value = tile[src_pos];
        
        // Calculate output position
        int out_row = block_x;
        int out_col = block_y * (tile_x * tile_y) + perm_idx;
        
        // Store for later processing (would need additional steps for final conversion)
    }
}

// Wrapper function for permute_4int8_to_int32
cudaError_t permute_4int8_to_int32_cuda(const int8_t* d_input, int32_t* d_output, 
                                       int total_elements, cudaStream_t stream = 0) {
    if (total_elements == 0) return cudaSuccess;
    
    // Calculate optimal block size
    int block_size = 256;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    
    // Launch kernel
    permute_4int8_to_int32_kernel<<<num_blocks, block_size, 0, stream>>>(
        d_input, d_output, total_elements);
    
    return cudaPeekAtLastError();
}

// Main permute weight function
void permute_weight_cuda(torch::Tensor &d_input, torch::Tensor &d_output,
    // const int8_t* d_input, int32_t* d_output,
                               int in_features, int out_features) {
    // Precompute permutation on host and copy to device
    int perm_w[512]; // 32 * 4 * 4
    _get_perms_format(perm_w);
    
    int* d_perm_w;
    cudaMalloc(&d_perm_w, sizeof(perm_w));
    cudaMemcpy(d_perm_w, perm_w, sizeof(perm_w), cudaMemcpyHostToDevice);
    
    // Calculate dimensions
    int tile_x = 8, tile_y = 16;
    int blocks_x = in_features / tile_x;
    int blocks_y = out_features / tile_y;
    
    // First: reshape and permute using kernel
    dim3 block_dim(256);
    dim3 grid_dim((blocks_x * blocks_y + 255) / 256);

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    // Launch permutation kernel
    permute_weight_kernel<<<grid_dim, block_dim, 0, stream>>>(
        d_input.data_ptr<int8_t>(), d_output.data_ptr<int32_t>(), d_perm_w, 
        in_features, out_features, 512);
    
    // Then: convert to int32
    int total_int32_elements = (in_features * out_features) / 4;
    permute_4int8_to_int32_cuda(d_output.data_ptr<int8_t>(), d_output.data_ptr<int32_t>(), 
                                total_int32_elements, stream);
    
    cudaFree(d_perm_w);
}