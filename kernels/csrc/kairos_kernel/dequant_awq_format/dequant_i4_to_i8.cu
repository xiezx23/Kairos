// Author: Zexi Xie.

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "fast_dequant.h"

// if q_factor(scales, scaled_zeros).shape == [K/group_size, N]
#define Q_FACTOR_T

__device__ __forceinline__ void deq_uint4_from_int16(
        const uint16_t packed, const __half scale, const __half scaled_zero,
        __half& out1, __half& out2, __half& out3, __half& out4) 
{
    ushort val4 = (packed >> 12) & 0x0F;
    ushort val3 = (packed >> 8) & 0x0F;
    ushort val2 = (packed >> 4) & 0x0F;
    ushort val1 = packed & 0x0F;

    __half2 val12 = __halves2half2(__ushort2half_rn(val1), __ushort2half_rn(val2));
    __half2 val34 = __halves2half2(__ushort2half_rn(val3), __ushort2half_rn(val4));
    __half2 scale2 = __half2half2(scale);
    __half2 scaled_zero2 = __half2half2(scaled_zero);
    
    // __hfma2 = __hadd2(__hmul2(val, scale), scaled_zero);
    __half2 result12 = __hfma2(val12, scale2, scaled_zero2);
    __half2 result34 = __hfma2(val34, scale2, scaled_zero2);

    out1 = result12.x; out2 = result12.y;
    out3 = result34.x; out4 = result34.y;
}

__device__ __forceinline__ uint16_t load_uint16(const uint16_t* ptr) {
    return *ptr;
}

__global__ void dequant_interleaved_int4_to_int8_kernel(
        int8_t* __restrict__ output,
        const uint16_t* __restrict__ input,
        const __half* __restrict__ scale,
        const __half* __restrict__ scaled_zero,
        const int N,
        const int K,
        const int group_size,
        const int interleave,
        const int kstride) {
    // input: [N/4, K/64, 4, packed(64/4)]
    // every block will solve the dequatization for 8 nums * 4 rows each turn.
    const int row_block = blockIdx.x;
    const int tid = threadIdx.x;
    
    const int num_k_blocks = K / kstride;
    const int num_groups = K / group_size;
    
    extern __shared__ __half shared_data[];
    __half* shared_scale = shared_data;
    __half* shared_scaled_zero = shared_data + num_groups*interleave;
    
    // loading scale and scaled_zero to share mem for interleave rows.
    #pragma unroll
    for (int i = tid; i < num_groups * interleave; i += blockDim.x) {
        #ifndef Q_FACTOR_T  // share(scales, scaled_zeros).shape == [interleave, K/group_size]
        const int row_in_block = i / num_groups;
        const int group_idx    = i % num_groups;
        const int actual_row   = row_block * interleave + row_in_block;
        if (actual_row < N) {
            const int gidx = actual_row * num_groups + group_idx;
            shared_scale[i]       = scale[gidx];
            shared_scaled_zero[i] = scaled_zero[gidx];
        }
        #else               // share(scales, scaled_zeros).shape == [K/group_size, interleave]
        const int row_in_block = i % interleave;
        const int group_idx    = i / interleave;
        const int actual_row   = row_block * interleave + row_in_block;
        if (actual_row < N) {
            const int gidx = group_idx * N + actual_row;
            shared_scale[i]       = scale[gidx];
            shared_scaled_zero[i] = scaled_zero[gidx];
        }
        #endif
    }
    __syncthreads();
    
    const int elems_per_thread = 8; // 每个线程每次每行处理8个输出元素（2个int16）
    const int solve_end = K / elems_per_thread;
    
    for (int cur_tid = tid; cur_tid < solve_end; cur_tid += blockDim.x) {
        const int k_idx = cur_tid * elems_per_thread;
        const int block_kidx = k_idx / (kstride);
        const int inner_kidx = k_idx % kstride;
        const int elems_bidx = ((k_idx % 32) / 8) * 2;
        const uint16_t* input_base_ptr = input + row_block * K + block_kidx * kstride + inner_kidx/4;
        // 处理interleave行
        #pragma unroll
        for (int row_in_block = 0; row_in_block < interleave; row_in_block++) {
            // input: [N/4, K/64, 4, packed(64/4)]
            // const uint16_t* input_ptr = input + row_block * K + block_kidx * kstride + row_in_block*(kstride/4) + inner_kidx/4;
            const uint16_t* input_ptr = input_base_ptr + row_in_block*(kstride/4);
            int act_row = row_block * interleave + row_in_block;
            int8_t* output_ptr = output + act_row * K;
            // get corresponding s and z
            const int group_id = k_idx / group_size;
            #ifndef Q_FACTOR_T  // q_factor(scales, scaled_zeros).shape == [N, K/group_size]
            const int gidx = row_in_block * num_groups + group_id;
            #else               // q_factor(scales, scaled_zeros).shape == [K/group_size, N]
            const int gidx = row_in_block + group_id * interleave;
            #endif
            const __half s = shared_scale[gidx];
            const __half z = shared_scaled_zero[gidx];
            /* fast dequantization */
            // half half_weight_buffer[8];
            // dequantize_to_half8(*reinterpret_cast<const half2*>(input_ptr), reinterpret_cast<uint4 *>(half_weight_buffer));
            // __half2 w02 = *reinterpret_cast<__half2*>(half_weight_buffer);
            // __half2 w46 = *reinterpret_cast<__half2*>(half_weight_buffer + 2);
            // __half2 w13 = *reinterpret_cast<__half2*>(half_weight_buffer + 4);
            // __half2 w57 = *reinterpret_cast<__half2*>(half_weight_buffer + 6);
            // w02 = __hfma2(w02, __half2half2(s), __half2half2(z));
            // w46 = __hfma2(w46, __half2half2(s), __half2half2(z));
            // w13 = __hfma2(w13, __half2half2(s), __half2half2(z));
            // w57 = __hfma2(w57, __half2half2(s), __half2half2(z));
            // output_ptr[k_idx+0] = w02.x; output_ptr[k_idx+2] = w02.y;
            // output_ptr[k_idx+1] = w13.x; output_ptr[k_idx+3] = w13.y;
            // output_ptr[k_idx+4] = w46.x; output_ptr[k_idx+6] = w46.y;
            // output_ptr[k_idx+5] = w57.x; output_ptr[k_idx+7] = w57.y;

            // get 2 packed int16 val (2*4 int4 data)
            u_int16_t packed_val1 = load_uint16(input_ptr);
            u_int16_t packed_val2 = load_uint16(input_ptr + 1);
            // unpack 8 value: [0, 2, 4, 6, 1, 3, 5, 7] => [0, 1, 2, 3, 4, 5, 6, 7]
            __half val0, val1, val2, val3, val4, val5, val6, val7;
            deq_uint4_from_int16(packed_val1, s, z, val0, val2, val4, val6);
            deq_uint4_from_int16(packed_val2, s, z, val1, val3, val5, val7);
            // output_ptr[k_idx+0] = val0; output_ptr[k_idx+1] = val1;
            // output_ptr[k_idx+2] = val2; output_ptr[k_idx+3] = val3;
            // output_ptr[k_idx+4] = val4; output_ptr[k_idx+5] = val5;
            // output_ptr[k_idx+6] = val6; output_ptr[k_idx+7] = val7;

            // write value: [0, 1, 8, 9, 16, 17, ...] =>[0, 1, 2, 3, 4, 5, 6, 7...]
            const int _idx = k_idx - elems_bidx * 4;
            output_ptr[_idx+ 0+elems_bidx] = (int8_t)val0; output_ptr[_idx+ 1+elems_bidx] = (int8_t)val1;
            output_ptr[_idx+ 8+elems_bidx] = (int8_t)val2; output_ptr[_idx+ 9+elems_bidx] = (int8_t)val3;
            output_ptr[_idx+16+elems_bidx] = (int8_t)val4; output_ptr[_idx+17+elems_bidx] = (int8_t)val5;
            output_ptr[_idx+24+elems_bidx] = (int8_t)val6; output_ptr[_idx+25+elems_bidx] = (int8_t)val7;
        }
    }
}

void dequant_interleaved_int4_to_int8(
        torch::Tensor &output,      // [N, K], dtype=torch.int8
        torch::Tensor &input,       // [N/interleave, K], dtype=torch.int16  
        // [N, K/group_size] or [K/group_size, N], dtype=torch.float16
        torch::Tensor &scale,
        torch::Tensor &scaled_zero,
        int group_size) {
    const int interleave=4;
    const int kstride=64;
    const int N = output.size(0);
    const int K = output.size(1);
    
    // TORCH_CHECK(K % kstride == 0,    "K must be divisible by kstride");
    // TORCH_CHECK(N % interleave == 0, "N must be divisible by interleave");
    // TORCH_CHECK(K % group_size == 0, "K must be divisible by group_size");
    
    const int num_groups = K / group_size;
    const size_t shared_mem_size = 2 * num_groups * interleave * sizeof(__half);
    
    const dim3 grid(N / interleave);
    const dim3 block(128);
    
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    dequant_interleaved_int4_to_int8_kernel<<<grid, block, shared_mem_size, stream>>>(
        reinterpret_cast<int8_t*>(output.data_ptr()),
        reinterpret_cast<const uint16_t*>(input.data_ptr<int16_t>()),
        reinterpret_cast<const __half*>(scale.data_ptr()),
        reinterpret_cast<const __half*>(scaled_zero.data_ptr()),
        N, K, group_size, interleave, kstride);
}