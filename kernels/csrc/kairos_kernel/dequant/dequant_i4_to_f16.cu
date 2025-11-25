#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "dequant_i4_to_f16.h"

__device__ __forceinline__ void uint4_to_fp16(
        const uint8_t packed, 
        const __half scale, 
        const __half zero, 
        __half& out1, 
        __half& out2) {
    ushort high4 = (packed >> 4) & 0x0F;
    ushort low4 = packed & 0x0F;

    // __half val1 = __ushort2half_rn(high4);
    // __half val2 = __ushort2half_rn(low4);
    // out1 = __hmul(__hsub(val1, zero), scale);
    // out2 = __hmul(__hsub(val2, zero), scale);

    __half2 val = __halves2half2(__ushort2half_rn(high4), __ushort2half_rn(low4));
    __half2 zero2 = __half2half2(zero);
    __half2 scale2 = __half2half2(scale);
    __half2 out = __hmul2(__hsub2(val, zero2), scale2);
    out1 = out.x;
    out2 = out.y;
}

__device__ __forceinline__ uint2 load_uint2(const uint8_t* ptr) {
    uint2 val;
    val.x = ptr[0];
    val.y = ptr[1];
    return val;
}

__device__ __forceinline__ void store_half2(__half* ptr, const __half2 val) {
    ptr[0] = val.x;
    ptr[1] = val.y;
}

__global__ void dequant_uint4_to_fp16_kernel(
        __half* __restrict__ output,
        const uint8_t* __restrict__ input,
        const __half* __restrict__ scale,
        const __half* __restrict__ zero,
        const int N, 
        const int K,
        const int group_size) {
    
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int num_groups = K / group_size;
    
    extern __shared__ __half shared_data[];
    __half* shared_scale = shared_data;
    __half* shared_zero = shared_data + num_groups;
    
    for (int i = tid; i < num_groups; i += blockDim.x) {
        shared_scale[i] = scale[row * num_groups + i];
        shared_zero[i] = zero[row * num_groups + i];
    }
    __syncthreads();
    
    const int elems_per_thread  = 32 / 4; // each theard block -> 8 output elements
    const int solve_end         = K / elems_per_thread;
    for (int cur_tid = tid; cur_tid < solve_end; cur_tid += blockDim.x) {
        const int start_idx = cur_tid * elems_per_thread;
        const int end_idx = min(start_idx + elems_per_thread, K);
        const uint8_t* input_ptr = input + row * (K / 2);
        __half* output_ptr = output + row * K;
        for (int idx = start_idx; idx < end_idx; idx += 4) {
            // if (idx + 3 >= K) { 
            //     for (int i = idx; i < min(idx + 4, K); i++) {
            //         const int group_id = i / group_size;
            //         const __half s = shared_scale[group_id];
            //         const __half z = shared_zero[group_id];
            //         const int input_idx = i / 2;
            //         const uint8_t packed_val = input_ptr[input_idx];
            //         __half val1, val2;
            //         uint4_to_fp16(packed_val, s, z, val1, val2);
            //         if (i % 2 == 0)  output_ptr[i] = val1;
            //         else  output_ptr[i] = val2;
            //     }
            //     break;
            // }
            const int group_id1 = idx / group_size;
            const int group_id2 = (idx + 2) / group_size;
            const __half s1 = shared_scale[group_id1];
            const __half z1 = shared_zero[group_id1];
            const __half s2 = (group_id1 == group_id2) ? s1 : shared_scale[group_id2];
            const __half z2 = (group_id1 == group_id2) ? z1 : shared_zero[group_id2];
            // 加载2个字节的输入数据（包含4个UINT4值）
            const uint2 packed_vals = load_uint2(input_ptr + idx / 2);
            // 处理第一个字节（2个UINT4值）
            __half val1, val2;
            uint4_to_fp16(packed_vals.x, s1, z1, val1, val2);
            output_ptr[idx] = val1;
            output_ptr[idx + 1] = val2;
            // 处理第二个字节（2个UINT4值）
            uint4_to_fp16(packed_vals.y, s2, z2, val1, val2);
            output_ptr[idx + 2] = val1;
            output_ptr[idx + 3] = val2;
        }
    }
}

void dequant_int4_to_fp16(torch::Tensor &output,// [N, K], dtype=torch.float16
                        torch::Tensor &input,   // [N, K/2], dtype=torch.int8
                        torch::Tensor &scale,   // [N, K/group_size], dtype=torch.float16
                        torch::Tensor &zero,    // [N, K/group_size], dtype=torch.float16
                        int group_size) {
    // TORCH_CHECK(output.is_contiguous(), "Output tensor must be contiguous");
    // TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    // TORCH_CHECK(scale.is_contiguous(), "Scale tensor must be contiguous");
    // TORCH_CHECK(zero.is_contiguous(), "Zero tensor must be contiguous");
    // TORCH_CHECK(output.scalar_type() == torch::kFloat16, "Output must be FP16");
    // TORCH_CHECK(input.scalar_type() == torch::kInt8, "Input must be INT8");
    // TORCH_CHECK(scale.scalar_type() == torch::kFloat16, "Scale must be FP16");
    // TORCH_CHECK(zero.scalar_type() == torch::kFloat16, "Zero must be FP16");
    // TORCH_CHECK(group_size > 0, "Group size must be positive");
    
    const int N = output.size(0);
    const int K = output.size(1);
    // TORCH_CHECK(K % group_size == 0, "K must be divisible by group_size");
    
    const int num_groups = K / group_size;
    
    const size_t shared_mem_size = 2 * num_groups * sizeof(__half);
    
    const dim3 grid(N);
    const dim3 block(256);
    
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    dequant_uint4_to_fp16_kernel<<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<__half*>(output.data_ptr()),
        reinterpret_cast<const uint8_t*>(input.data_ptr<int8_t>()),
        reinterpret_cast<const __half*>(scale.data_ptr()),
        reinterpret_cast<const __half*>(zero.data_ptr()),
        N, K, group_size);
}


void dequant_int4_to_fp16_stream(torch::Tensor &output,// [N, K], dtype=torch.float16
                        torch::Tensor &input,   // [N, K/2], dtype=torch.int8
                        torch::Tensor &scale,   // [N, K/group_size], dtype=torch.float16
                        torch::Tensor &zero,    // [N, K/group_size], dtype=torch.float16
                        int group_size,
                        uintptr_t stream_ptr) {
    
    const int N = output.size(0);
    const int K = output.size(1);
    const int num_groups = K / group_size;
    const size_t shared_mem_size = 2 * num_groups * sizeof(__half);

    // 计算每次启动的线程块数量
    int chunks = N;
    if (N > 4096) {
        chunks = (N + 5) / 8; // 等价于ceil(N/8)
    } else {
        chunks = N;
    }

    const dim3 grid(chunks);
    const dim3 block(256);

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

    // 分批次启动内核
    for (int start = 0; start < N; start += chunks) {
        int current_chunk = std::min(chunks, N - start);
        const dim3 grid(current_chunk);
        // 计算当前批次的指针偏移
        __half* output_ptr = reinterpret_cast<__half*>(output.data_ptr()) + start * K;
        const uint8_t* input_ptr = reinterpret_cast<const uint8_t*>(input.data_ptr<int8_t>()) + start * (K/2);
        const __half* scale_ptr = reinterpret_cast<const __half*>(scale.data_ptr()) + start * num_groups;
        const __half* zero_ptr = reinterpret_cast<const __half*>(zero.data_ptr()) + start * num_groups;

        dequant_uint4_to_fp16_kernel<<<grid, block, shared_mem_size, stream>>>(
            output_ptr, input_ptr,
            scale_ptr, zero_ptr,
            current_chunk, // 当前批次的实际大小
            K, group_size);
    }
}