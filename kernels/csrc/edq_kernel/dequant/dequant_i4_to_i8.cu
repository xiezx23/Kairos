// Author: Zexi Xie.

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cassert>
#include <stdint.h>
#include <float.h>
#include <type_traits>
#include <cuda_fp16.h>

__global__ void dequant_kernel_i4_to_i8(
        int8_t * __restrict__ output,
        const int8_t * __restrict__ input,
        const int8_t * __restrict__ scale,
        const int8_t * __restrict__ zero,
        int N, int K,
        int group_size) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int num_groups = K / group_size;

    extern __shared__ int8_t shared_data[];
    int8_t *shared_scale = shared_data;
    int8_t *shared_zero = shared_data + num_groups;

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
}

void dequant_int4_to_int8(
        torch::Tensor &out,   // [N, K]
        torch::Tensor &input,   // [N, K/2]
        torch::Tensor &scale,   // [N, K/group_size]
        torch::Tensor &zero,    // [N, K/group_size]
        int group_size) {
    // assert(out.is_contiguous());
    // assert(input.is_contiguous());
    // assert(scale.is_contiguous());
    // assert(zero.is_contiguous());
    // assert(group_size > 0);
    int N = out.size(0), K = out.size(1);
    // assert(K % group_size == 0);
    int num_groups = K / group_size;
    dim3 grid(N);
    dim3 block(256);
    size_t shared_mem_size = 2 * num_groups * sizeof(int8_t);
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    dequant_kernel_i4_to_i8<<<grid, block, shared_mem_size, stream>>>(
        out.data_ptr<int8_t>(),
        input.data_ptr<int8_t>(),
        scale.data_ptr<int8_t>(),
        zero.data_ptr<int8_t>(),
        N, K, group_size
    );
}

void dequant_int4_to_int8_stream(torch::Tensor &out,   // [N, K]
                        torch::Tensor &input,   // [N, K/2]
                        torch::Tensor &scale,   // [N, K/group_size]
                        torch::Tensor &zero,    // [N, K/group_size]
                        int group_size,
                        uintptr_t stream_ptr) {
    // assert(out.is_contiguous());
    // assert(input.is_contiguous());
    // assert(scale.is_contiguous());
    // assert(zero.is_contiguous());
    // assert(group_size > 0);
    int N = out.size(0), K = out.size(1);
    // assert(K % group_size == 0);
    int num_groups = K / group_size;
    dim3 grid(N);
    dim3 block(256);
    size_t shared_mem_size = 2 * num_groups * sizeof(int8_t);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    dequant_kernel_i4_to_i8<<<grid, block, shared_mem_size, stream>>>(
        out.data_ptr<int8_t>(),
        input.data_ptr<int8_t>(),
        scale.data_ptr<int8_t>(),
        zero.data_ptr<int8_t>(),
        N, K, group_size
    );
}


// __global__ void dequant_kernel( int8_t * __restrict__ output,
//                                 const int8_t * __restrict__ input,
//                                 const int8_t * __restrict__ scale,
//                                 const int8_t * __restrict__ zero,
//                                 int N, int K,
//                                 int group_size,
//                                 int group_num_per_row) {
//     const int group_id  = threadIdx.x;
//     const int row    = blockIdx.x;
//     const int8_t mask   = 0b00001111;
//     #pragma unroll
//     for (int gid = group_id; gid < group_num_per_row; gid += blockDim.x) {
//         const int8_t s  = scale[row*group_num_per_row + gid];
//         const int8_t z    = zero[row*group_num_per_row + gid];
//         const int data_idx  = row*K + gid*group_size;
//         #pragma unroll
//         for (int i = 0; i < group_size; i += 2) {
//             const int8_t input_packed_data = input[(data_idx+i) / 2];
//             output[data_idx+i]   = (((input_packed_data >> 4) & mask) - z ) * s;
//             output[data_idx+i+1] = ((input_packed_data & mask) - z ) * s;
//         }
//     }
// }

// void dequant_int4_to_int8(torch::Tensor &out,   // [N, K]
//                         torch::Tensor &input,   // [N, K/2]
//                         torch::Tensor &scale,   // [N, K/group_size]
//                         torch::Tensor &zero,    // [N, K/group_size]
//                         int group_size) {
//     assert(out.is_contiguous());
//     assert(input.is_contiguous());
//     assert(group_size > 0);
//     int N = out.size(0), K = out.size(1);
//     int group_num_per_row = K / group_size;
//     dim3 grid(N);
//     dim3 block(std::min(group_num_per_row, 1024));
//     const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
//     dequant_kernel<<<grid, block, 0, stream>>>(out.data_ptr<int8_t>(),
//         input.data_ptr<int8_t>(), scale.data_ptr<int8_t>(),
//         zero.data_ptr<int8_t>(), N, K, group_size, group_num_per_row);
// }
