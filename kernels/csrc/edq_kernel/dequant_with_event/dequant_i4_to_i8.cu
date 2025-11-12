// Author: Zexi Xie.

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cassert>
#include <stdint.h>
#include <float.h>
#include <type_traits>
#include <cuda_fp16.h>

__global__ void dequant_kernel_i4_to_i8_new(
        int8_t * __restrict__ output,
        const int8_t * __restrict__ input,
        const int8_t * __restrict__ scale,
        const int8_t * __restrict__ zero,
        int N, int K,
        int group_size) {
    const int tid = threadIdx.x;
    const int num_groups = K / group_size;

    extern __shared__ int8_t shared_data[];
    int8_t *shared_scale = shared_data;
    int8_t *shared_zero = shared_data + num_groups;

    for (int row = blockIdx.x; row < N; row += gridDim.x) {
        #pragma unroll
        for (int i = tid; i < num_groups; i += blockDim.x) {
            shared_scale[i] = scale[row * num_groups + i];
            shared_zero[i] = zero[row * num_groups + i];
        }
        __syncthreads();

        #pragma unroll
        for (int idx = tid; idx < K; idx += blockDim.x) {
            int gidx = idx / group_size;
            int8_t s = shared_scale[gidx];
            int8_t z = shared_zero[gidx];
            int input_idx = (row * K + idx) >> 1;
            int8_t input_byte = input[input_idx];
            int8_t val;
            val = idx % 2 ? (input_byte & 0x0F) : ((input_byte >> 4) & 0x0F);
            // if (idx % 2 == 0) {
            //     val = (input_byte >> 4) & 0x0F;
            // } else {
            //     val = input_byte & 0x0F;
            // }
            output[row * K + idx] = (val - z) * s;
        }
        __syncthreads();
    }
}


void dequant_int4_to_int8_new_2(torch::Tensor &out,   // [N, K]
                        torch::Tensor &input,   // [N, K/2]
                        torch::Tensor &scale,   // [N, K/group_size]
                        torch::Tensor &zero,    // [N, K/group_size]
                        int group_size,
                        int grid_size) {
    // assert(out.is_contiguous());
    // assert(input.is_contiguous());
    // assert(scale.is_contiguous());
    // assert(zero.is_contiguous());
    // assert(group_size > 0);
    int N = out.size(0), K = out.size(1);
    // assert(K % group_size == 0);
    int num_groups = K / group_size;
    dim3 grid(grid_size);
    dim3 block(256);
    size_t shared_mem_size = 2 * num_groups * sizeof(int8_t);
    dequant_kernel_i4_to_i8_new<<<grid, block, shared_mem_size>>>(
        out.data_ptr<int8_t>(),
        input.data_ptr<int8_t>(),
        scale.data_ptr<int8_t>(),
        zero.data_ptr<int8_t>(),
        N, K, group_size
    );
}

void dequant_int4_to_int8_new(
            torch::Tensor &out,     // [N, K]
            torch::Tensor &input,   // [N, K/2]
            torch::Tensor &scale,   // [N, K/group_size]
            torch::Tensor &zero,    // [N, K/group_size]
            int group_size,
            int grid_size) {
    // assert(out.is_contiguous());
    // assert(input.is_contiguous());
    // assert(scale.is_contiguous());
    // assert(zero.is_contiguous());
    // assert(group_size > 0);
    int N = out.size(0), K = out.size(1);
    // assert(K % group_size == 0);
    int num_groups = K / group_size;
    
    dim3 grid(grid_size);
    dim3 block(256);
    size_t shared_mem_size = 2 * num_groups * sizeof(int8_t);
    // cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    // const auto current_device = c10::cuda::current_device();
    // cudaStream_t stream = at::cuda::getStreamFromExternal(reinterpret_cast<cudaStream_t>(stream_ptr), current_device).stream();
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    dequant_kernel_i4_to_i8_new<<<grid, block, shared_mem_size, stream>>>(
        out.data_ptr<int8_t>(),
        input.data_ptr<int8_t>(),
        scale.data_ptr<int8_t>(),
        zero.data_ptr<int8_t>(),
        N, K, group_size
    );
}