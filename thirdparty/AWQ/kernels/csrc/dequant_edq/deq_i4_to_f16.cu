#include <cuda_fp16.h>
#include <stdio.h>
#include <torch/extension.h>
#include "../quantization_new/dequantize.cuh"

#define PACK_FACTOR 8
#define WARP_SIZE 32
#define MEM_ACCESS_SIZE 128

template <int GroupSize, int BlockSize, typename T>
__global__ void dequantize_kernel(
                    const uint32_t* weight, 
                    const T* scales, 
                    const T* zeros, 
                    T* output,
                    const int IC, 
                    const int OC) {
    const int kElemsPerThread = MEM_ACCESS_SIZE / 4; // 32 elements per thread
    const int kStride = 64;
    const int kThreadsNumPerTile = kStride / kElemsPerThread;
    
    static constexpr int kShuffleSize = 32;
    static constexpr int kShuffleBasicTile = 2;
    static constexpr int kShuffleContinous = 4;
    static constexpr int kShuffleStrided = 4;

    // Each block processes multiple output channels
    const int blk_row_offset = blockIdx.x * BlockSize;
    const int thd_row_offset = (threadIdx.x / kThreadsNumPerTile) % kShuffleContinous;
    const int act_k_offset = threadIdx.x / (kThreadsNumPerTile * kShuffleContinous) * kStride
                               + (threadIdx.x % kThreadsNumPerTile) * kElemsPerThread;
    const int group_offset = act_k_offset / GroupSize;

    // Pointers to input data
    const uint32_t* blk_weight_ptr = weight + blk_row_offset * IC / PACK_FACTOR;
    const T* scale_ptr = scales + blk_row_offset + thd_row_offset + group_offset * OC;
    const T* zeros_ptr = zeros + blk_row_offset + thd_row_offset + group_offset * OC;
    
    // Output pointer - each block writes BlockSize output channels
    T* output_ptr = output + blk_row_offset * IC;

    const int act_forward_step = BlockSize * kElemsPerThread / kShuffleContinous;
    const int scale_forward_step = act_forward_step / GroupSize * OC;

    // Process all input channels
    for (int kk = threadIdx.x * kElemsPerThread; kk < IC * kShuffleContinous; kk += BlockSize * kElemsPerThread){
        // Process each output channel in this block
        for (int oc_idx = 0; oc_idx < BlockSize; ++oc_idx){
            uint32_t local_qweights[MEM_ACCESS_SIZE / 32];
            T half_weight_buffer[kElemsPerThread];
            T dequantized_weight[kElemsPerThread];

            // Load packed weights (4-bit)
            *((float4*)(local_qweights)) = 
                *((float4*)(blk_weight_ptr + (oc_idx * kShuffleContinous * IC + kk) / PACK_FACTOR));
            
            // Load scale and zero for this output channel and group
            T local_scale = *(scale_ptr + oc_idx * kShuffleContinous);
            T local_zero = *(zeros_ptr + oc_idx * kShuffleContinous);

            // Convert int4 to fp16
            #pragma unroll
            for (int i = 0; i < MEM_ACCESS_SIZE / 32; ++i) {
                dequantize_s4_to_fp16x2<T>(*reinterpret_cast<half2*>(local_qweights + i), 
                            reinterpret_cast<uint4*>(half_weight_buffer + i * PACK_FACTOR));
            }
            // Dequantize and shuffle elements to match the original format
            #pragma unroll
            for (int i = 0; i < kShuffleContinous; ++i) {
                #pragma unroll
                for (int j = 0; j < kShuffleStrided; ++j) {
                    half2 w = *reinterpret_cast<half2*>(
                        half_weight_buffer + (i + j * kShuffleContinous) * kShuffleBasicTile
                    );
                    // Apply dequantization: weight = quantized * scale + zero
                    w = __hfma2(w, __half2half2(local_scale), __half2half2(local_zero));
                    // Store dequantized weights in the correct positions
                    dequantized_weight[(i * kShuffleStrided + j) * kShuffleBasicTile + 0] = w.x;
                    dequantized_weight[(i * kShuffleStrided + j) * kShuffleBasicTile + 1] = w.y;
                }
            }
            // Write dequantized weights to global memory
            const int output_offset = oc_idx * IC + (kk / kShuffleContinous);
            #pragma unroll
            for (int i = 0; i < kElemsPerThread; ++i) {
                if (output_offset + i < IC)  // Boundary check
                    output_ptr[output_offset + i] = dequantized_weight[i];
            }
        }
        // Move pointers forward
        scale_ptr += scale_forward_step;
        zeros_ptr += scale_forward_step;
    }
}

/*
Dequantize packed 4-bit weights back to fp16 (PyTorch interface).

Args:
  _kernel: int tensor of shape [OC, IC // 8] - packed 4-bit weights
  _scaling_factors: tensor of shape [OC, IC // G] - scale factors
  _zeros: tensor of shape [OC, IC // G] - zero points
  IC: input channels
  OC: output channels  
  group_size: group size for quantization

Returns:
  dequantized_weights: tensor of shape [OC, IC] - dequantized fp16 weights
*/
torch::Tensor dequantize_i4_to_f16_cuda(
    torch::Tensor _kernel,
    torch::Tensor _scaling_factors,
    torch::Tensor _zeros,
    int IC,
    int OC,
    int group_size)
{
    auto data_type = _scaling_factors.scalar_type();
    TORCH_CHECK(_zeros.scalar_type() == data_type);
    TORCH_CHECK(data_type == torch::kFloat16, "Only fp16 are supported");
    // Create output tensor [OC, IC]
    auto options = torch::TensorOptions().dtype(_scaling_factors.dtype()).device(_scaling_factors.device());
    at::Tensor _dequantized_weights = torch::empty({OC, IC}, options);

    auto kernel = reinterpret_cast<uint32_t*>(_kernel.data_ptr());
    auto scaling_factors = reinterpret_cast<half*>(_scaling_factors.data_ptr());
    auto zeros = reinterpret_cast<half*>(_zeros.data_ptr());
    auto dequantized_weights = reinterpret_cast<half*>(_dequantized_weights.data_ptr());

    static constexpr int BLOCK_SIZE = 16;  // Output channels per block
    static constexpr int NUM_THREADS = 256;

    // Calculate grid size - each block processes BLOCK_SIZE output channels
    dim3 num_blocks((OC + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 num_threads(NUM_THREADS);

    dequantize_kernel<128, BLOCK_SIZE><<<num_blocks, num_threads>>>(
        kernel, scaling_factors, zeros, dequantized_weights, IC, OC
    );

    return _dequantized_weights;
}