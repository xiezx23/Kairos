// Inspired by AWQ.
// Author: Junwei Chen.

#include <cuda_fp16.h>

// #include <cute/tensor.hpp>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

#include "w8a16_ws_gemv.h"
#include "utils.h"

// Reduce sum within the warp
template <int Batch, int CTA_N, int Thread>
__forceinline__ __device__ static void warp_reduce(float* psum, float (*out_smem)[Batch]) {
    float sum_fp32[Batch];
    #pragma unroll
    for (int i = 0; i < Batch; i++)
        sum_fp32[i] = psum[i];

    // warp内reduce
    #pragma unroll
    for (int i = 0; i < Batch; i++) {
        for (int thr = (Thread >> 1); thr > 0; thr = (thr >> 1)) {
            sum_fp32[i] += __shfl_xor_sync(0xffffffff, sum_fp32[i], thr);
        }
    }

    __syncthreads();

    // R2S
    int lane_idx = threadIdx.x;
    int warp_idx = threadIdx.y;
    int WARP_K = 32 / CTA_N;
    // maybe WARP_K 
    if (lane_idx % WARP_K == 0) {
        #pragma unroll
        for (int i = 0; i < Batch; i++)
            out_smem[warp_idx * CTA_N + lane_idx / WARP_K][i] = sum_fp32[i];

    }
    
    __syncthreads();
}


template <int CTA_N, int Batch, int BlockSize>
__global__ void w8a16_ws_gemv(
    half* A,
    int8_t* B,
    half* C,
    half* B_scale,
    const int N,
    const int K
) {
    // 每个线程处理128b / 8b = 16个数据
    static constexpr int kElemsPerThread = 128 / 8;
    static constexpr int WARP_K = 32 / CTA_N; // 一个warp有多少个在k维度上处理
    static constexpr int NUM_WARP = BlockSize / 32;
    static constexpr int ALL_WARP_K = NUM_WARP * WARP_K;

    const int tidx = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_idx = threadIdx.y;
    const int lane_idx = threadIdx.x;

    // mem in reg
    // act, scale, qweight, dqweight, psum
    half A_reg[kElemsPerThread];
    int8_t B_ori_reg[kElemsPerThread];
    half B_dq_reg[kElemsPerThread];
    half2 B_scale_reg;

    // 这里用half还是float会影响精度，但精度损失与gemm相当
    // 精度损失主要在K处，K越大损失越大
    // half psum[Batch];
    float psum[Batch];
    for (int i = 0; i < Batch; i++)
        psum[i] = 0.0f;

    // mem in smem
    // out_smem(in difference warp)
    __shared__ float out_smem[NUM_WARP * CTA_N][Batch];

    // Offset, Ptr and Stride
    const int k_offset = (warp_idx * WARP_K + lane_idx % WARP_K) * kElemsPerThread;// TODO: 优化mod
    const int n_block_offset = blockIdx.x * CTA_N;
    const int n_thread_offset = lane_idx / WARP_K;
    
    // Thread Layout in B(CTA_N == 4):
    // T0 T8 T16 T24
    // T1 T9 T17 T25
    // ...
    // T沿着K维处理128b

    half* A_ptr = A + k_offset;
    int8_t* B_ptr = B + (n_block_offset + n_thread_offset) * K + k_offset;
    half* B_scale_ptr = B_scale + n_block_offset + n_thread_offset;

    const int k_stride = ALL_WARP_K * kElemsPerThread;

    // load scale
    B_scale_reg = __half2half2(*(B_scale_ptr));
    
#if 0
__syncthreads();
if (tidx == 9 && blockIdx.x == 0) {
    printf("k_offset %d, n_block_offset: %d, n_thread_offset: %d, k_stride: %d\n", k_offset, n_block_offset, n_thread_offset, k_stride);
    printf("scale: %.4f\n", static_cast<float>(B_scale_reg.x));
    printf("A start: %.4f\n", static_cast<float>(*(A_ptr)));
    printf("B start: %d\n", static_cast<int>(*(B_ptr)));
}
#endif

    // main loop
    const int NUM_K = K / k_stride;
    EdqFastNumericArrayConverter<half, int8_t, 4> convert_i8tofp16;

    for (int k = 0; k < NUM_K; k++) {
        // G2R B, 一次load 16 x int8, 共128b
        *(reinterpret_cast<float4*>(B_ori_reg)) = *(reinterpret_cast<float4*>(B_ptr + k * k_stride));

        // Type Convert, 一次4byte
        #pragma unroll
        for (int i = 0; i < 128 / 32; i++) {
            *(reinterpret_cast<float2*>(B_dq_reg) + i) = convert_i8tofp16(*(reinterpret_cast<uint32_t*>(B_ori_reg) + i));
        }

        // Dequantization
        half2* B_dq_base_ptr = reinterpret_cast<half2*>(B_dq_reg);
        #pragma unroll
        for (int i = 0; i < kElemsPerThread / 2; i++) {
            half2* B_dq_vec_ptr = B_dq_base_ptr + i;
            *B_dq_vec_ptr = __hmul2(*B_dq_vec_ptr, B_scale_reg);
        }

#if 0
__syncthreads();
if (tidx == 0 && blockIdx.x == 0) {
    printf("B ori 0: %.4f\n", static_cast<float>(B_ori_reg[0]));
    printf("B ori 1: %.4f\n", static_cast<float>(B_ori_reg[1]));
    printf("B ori 2: %.4f\n", static_cast<float>(B_ori_reg[2]));
    printf("B ori 3: %.4f\n", static_cast<float>(B_ori_reg[3]));
    printf("B ori 4: %.4f\n", static_cast<float>(B_ori_reg[4]));
    printf("B ori 5: %.4f\n", static_cast<float>(B_ori_reg[5]));
    printf("B ori 6: %.4f\n", static_cast<float>(B_ori_reg[6]));
    printf("B ori 7: %.4f\n", static_cast<float>(B_ori_reg[7]));

    printf("B dq 0: %.4f\n", static_cast<float>(B_dq_reg[0]));
    printf("B dq 1: %.4f\n", static_cast<float>(B_dq_reg[1]));
    printf("B dq 2: %.4f\n", static_cast<float>(B_dq_reg[2]));
    printf("B dq 3: %.4f\n", static_cast<float>(B_dq_reg[3]));
    printf("B dq 4: %.4f\n", static_cast<float>(B_dq_reg[4]));
    printf("B dq 5: %.4f\n", static_cast<float>(B_dq_reg[5]));
    printf("B dq 6: %.4f\n", static_cast<float>(B_dq_reg[6]));
    printf("B dq 7: %.4f\n", static_cast<float>(B_dq_reg[7]));
}
#endif

        // Compute
        #pragma unroll
        for (int bs = 0; bs < Batch; bs++) {
            // G2R A, 一次load 8 x half, 共128b
            half* A_load_ptr = A_ptr + bs * K + k * k_stride;
            #pragma unroll
            for (int i = 0; i < kElemsPerThread / 8; i++)
                *(reinterpret_cast<float4*>(A_reg + i * 8)) = *(reinterpret_cast<float4*>(A_load_ptr + i * 8));
            
            // FMA
            half2* A_base_ptr = reinterpret_cast<half2*>(A_reg);

#if 0
__syncthreads();
if (tidx == 0 && blockIdx.x == 0) {
    printf("A reg 0: %.4f\n", static_cast<float>(A_reg[0]));
    printf("A reg 1: %.4f\n", static_cast<float>(A_reg[1]));
    printf("A reg 2: %.4f\n", static_cast<float>(A_reg[2]));
    printf("A reg 3: %.4f\n", static_cast<float>(A_reg[3]));
    printf("A reg 4: %.4f\n", static_cast<float>(A_reg[4]));
    printf("A reg 5: %.4f\n", static_cast<float>(A_reg[5]));
    printf("A reg 6: %.4f\n", static_cast<float>(A_reg[6]));
    printf("A reg 7: %.4f\n", static_cast<float>(A_reg[7]));
    printf("A reg 8: %.4f\n", static_cast<float>(A_reg[8]));
    printf("A reg 9: %.4f\n", static_cast<float>(A_reg[9]));
    printf("A reg 10: %.4f\n", static_cast<float>(A_reg[10]));
    printf("A reg 11: %.4f\n", static_cast<float>(A_reg[11]));
    printf("A reg 12: %.4f\n", static_cast<float>(A_reg[12]));
    printf("A reg 13: %.4f\n", static_cast<float>(A_reg[13]));
    printf("A reg 14: %.4f\n", static_cast<float>(A_reg[14]));
    printf("A reg 15: %.4f\n", static_cast<float>(A_reg[15]));
}
#endif

            #pragma unroll
            for (int i = 0; i < kElemsPerThread / 2; i++) {
                psum[bs] += __half2float(
                    __hfma(
                        (A_base_ptr + i)->y,
                        (B_dq_base_ptr + i)->y,
                        __hfma((A_base_ptr + i)->x, (B_dq_base_ptr + i)->x, 0)
                    )
                );
                // half2 tmp = __hmul2(*(A_base_ptr + i), *(B_dq_base_ptr + i));
                // psum[bs] += __half2float(tmp.x + tmp.y);
            }
        }
    }

#if 0
__syncthreads();
// if (tidx == 0 && blockIdx.x == 0) {
//     printf("tidx: %d, psum: %.4f\n", tidx, static_cast<float>(psum[0]));
// }
// if (tidx == 1 && blockIdx.x == 0) {
//     printf("tidx: %d, psum: %.4f\n", tidx, static_cast<float>(psum[0]));
// }
// if (tidx == 2 && blockIdx.x == 0) {
//     printf("tidx: %d, psum: %.4f\n", tidx, static_cast<float>(psum[0]));
// }
// if (tidx == 3 && blockIdx.x == 0) {
//     printf("tidx: %d, psum: %.4f\n", tidx, static_cast<float>(psum[0]));
// }

// if (tidx == 4 && blockIdx.x == 0) {
//     printf("tidx: %d, psum: %.4f\n", tidx, static_cast<float>(psum[0]));
// }
// if (tidx == 5 && blockIdx.x == 0) {
//     printf("tidx: %d, psum: %.4f\n", tidx, static_cast<float>(psum[0]));
// }
// if (tidx == 6 && blockIdx.x == 0) {
//     printf("tidx: %d, psum: %.4f\n", tidx, static_cast<float>(psum[0]));
// }
// if (tidx == 7 && blockIdx.x == 0) {
//     printf("tidx: %d, psum: %.4f\n", tidx, static_cast<float>(psum[0]));
// }

if (tidx == 0 && blockIdx.x == 0) {
    printf("tidx: %d, psum: %.4f\n", tidx, static_cast<float>(psum[1]));
}
if (tidx == 1 && blockIdx.x == 0) {
    printf("tidx: %d, psum: %.4f\n", tidx, static_cast<float>(psum[1]));
}
if (tidx == 2 && blockIdx.x == 0) {
    printf("tidx: %d, psum: %.4f\n", tidx, static_cast<float>(psum[1]));
}
if (tidx == 3 && blockIdx.x == 0) {
    printf("tidx: %d, psum: %.4f\n", tidx, static_cast<float>(psum[1]));
}

if (tidx == 4 && blockIdx.x == 0) {
    printf("tidx: %d, psum: %.4f\n", tidx, static_cast<float>(psum[1]));
}
if (tidx == 5 && blockIdx.x == 0) {
    printf("tidx: %d, psum: %.4f\n", tidx, static_cast<float>(psum[1]));
}
if (tidx == 6 && blockIdx.x == 0) {
    printf("tidx: %d, psum: %.4f\n", tidx, static_cast<float>(psum[1]));
}
if (tidx == 7 && blockIdx.x == 0) {
    printf("tidx: %d, psum: %.4f\n", tidx, static_cast<float>(psum[1]));
}

// if (tidx == 8 && blockIdx.x == 0) {
//     printf("tidx: %d, psum: %.4f\n", tidx, static_cast<float>(psum[0]));
// }
// if (tidx == 32 && blockIdx.x == 0) {
//     printf("tidx: %d, psum: %.4f\n", tidx, static_cast<float>(psum[0]));
// }
// if (tidx == 33 && blockIdx.x == 0) {
//     printf("tidx: %d, psum: %.4f\n", tidx, static_cast<float>(psum[0]));
// }
// if (tidx == 40 && blockIdx.x == 0) {
//     printf("tidx: %d, psum: %.4f\n", tidx, static_cast<float>(psum[0]));
// }
// if (tidx == 64 && blockIdx.x == 0) {
//     printf("tidx: %d, psum: %.4f\n", tidx, static_cast<float>(psum[0]));
// }
// if (tidx == 65 && blockIdx.x == 0) {
//     printf("tidx: %d, psum: %.4f\n", tidx, static_cast<float>(psum[0]));
// }
// if (tidx == 72 && blockIdx.x == 0) {
//     printf("tidx: %d, psum: %.4f\n", tidx, static_cast<float>(psum[0]));
// }
#endif

    // Warp Reduce
    // warp内splik-k结果的Reduce
    warp_reduce<Batch, CTA_N, WARP_K>(psum, out_smem);

#if 0
if (tidx == 0 && blockIdx.x == 0) {
    printf("tidx: %d, out_smem: %.4f\n", tidx, static_cast<float>(out_smem[0][1]));
}
if (tidx == 4 && blockIdx.x == 0) {
    printf("tidx: %d, out_smem: %.4f\n", tidx, static_cast<float>(out_smem[1][1]));
}
#endif

    // warp之间通过Smem进行Reduce
    // 同时进行写回Gmem
    if (warp_idx == 0 && lane_idx % WARP_K == 0) {
        #pragma unroll
        for (int bs = 0; bs < Batch; bs++) {
            float acc = 0.0f;
            #pragma unroll
            for (int w = 0; w < NUM_WARP; w++)
                acc += out_smem[w * CTA_N + lane_idx / WARP_K][bs];
            C[bs * N + n_block_offset + n_thread_offset] = __float2half(acc);
        }
    }
}


void w8a16_ws_gemv_launch(
    half* qact,
    int8_t* qweight,
    half* out_feat,
    half* w_scale,
    const int batch_size,
    const int out_channel,
    const int in_channel
) {

    static constexpr int CTA_N = 8;
    static constexpr int NUM_WARP = 4;

    dim3 grid_dim(out_channel / CTA_N);
    dim3 block_dim(32, NUM_WARP);

    BATCH_SWITCH(batch_size, Batch, [&] {
        auto kernel = w8a16_ws_gemv<CTA_N, Batch, 32 * NUM_WARP>;

        kernel<<<grid_dim, block_dim>>>(
            qact, qweight, out_feat, 
            w_scale,
            out_channel, in_channel
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        // cudaDeviceSynchronize();
    });
}



// pybind接口
at::Tensor w8a16_ws_gemv_cuda(
    const at::Tensor &qact, // (batch_size, in_feat)
    const at::Tensor &qweight, // (out_feat, in_feat)
    const at::Tensor &w_scale, // (out_feat)
    std::optional<at::Tensor> &out_ // (batch_size, out_feat)
) {
    const int batch_size = qact.size(0);
    const int in_channel = qact.size(-1);
    const int out_channel = qweight.size(-2);
    TORCH_CHECK(qact.size(-1) == qweight.size(-1));
    TORCH_CHECK(qweight.size(-2) == w_scale.size(-1));
    // printf("w8a16_ws_gemv_cuda\n");

    at::Tensor out_feat;
    if (out_.has_value()) {
        out_feat = out_.value();
        TORCH_CHECK(out_feat.dtype() == at::kHalf);
        TORCH_CHECK(out_feat.size(-2) == qact.size(-2));
        TORCH_CHECK(out_feat.size(-1) == qweight.size(-2));
        TORCH_CHECK(out_feat.is_cuda(), "Out_feat must be on CUDA");
    }
    else {
        out_feat = torch::empty({batch_size, out_channel}, qact.options().dtype(at::kHalf));
    }

    auto qact_ptr = reinterpret_cast<half*>(qact.data_ptr());
    auto qweight_ptr = reinterpret_cast<int8_t*>(qweight.data_ptr());
    auto w_scale_ptr = reinterpret_cast<half*>(w_scale.data_ptr());
    auto out_feat_ptr = reinterpret_cast<half*>(out_feat.data_ptr());

    w8a16_ws_gemv_launch(
        qact_ptr, qweight_ptr, out_feat_ptr, 
        w_scale_ptr,
        batch_size, out_channel, in_channel
    );


    return out_feat;
}