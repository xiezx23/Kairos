// Inspired by AWQ.
// Author: Junwei Chen.

#include <cuda_fp16.h>

// #include <cute/tensor.hpp>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

#include "w4a16_wa_gemv.h"
#include "utils.h"

// Reduce sum within the warp
template <int Batch, int kElemsPerRowB, int Thread>
__forceinline__ __device__ static void warp_reduce(float* psum, float (*out_smem)[Batch * kElemsPerRowB]) {
    const int ResNum = Batch * kElemsPerRowB;
    // float sum_fp32[ResNum];
    // #pragma unroll
    // for (int i = 0; i < ResNum; i++)
    //     sum_fp32[i] = psum[i];

    // warp内reduce
    #pragma unroll
    for (int i = 0; i < ResNum; i++) {
        psum[i] += __shfl_xor_sync(0xffffffff, psum[i], 16);
        psum[i] += __shfl_xor_sync(0xffffffff, psum[i], 8);
        psum[i] += __shfl_xor_sync(0xffffffff, psum[i], 4);
        psum[i] += __shfl_xor_sync(0xffffffff, psum[i], 2);
        psum[i] += __shfl_xor_sync(0xffffffff, psum[i], 1);
        // for (int thr = (Thread >> 1); thr > 0; thr = (thr >> 1)) {
        //     sum_fp32[i] += __shfl_xor_sync(0xffffffff, sum_fp32[i], thr);
        // }
    }

    __syncthreads();

    // R2S
    int lane_idx = threadIdx.x;
    int warp_idx = threadIdx.y;
    // maybe WARP_K 
    if (lane_idx < kElemsPerRowB) {
        #pragma unroll
        for (int i = 0; i < Batch; i++)
            out_smem[warp_idx][i * kElemsPerRowB + lane_idx] = psum[i * kElemsPerRowB + lane_idx];
    }
    
    __syncthreads();
}


template <int Batch, int BlockSize>
__global__ void w4a16_wa_gemv(
    const half* A,
    const int* B,
    half* C,
    const half* B_scale,
    const half* B_scaled_zp,
    const int N,
    const int K
) {
    // Tensor Core Thread Layout in B:
    // T0 T4 T8 T12 T16 T20 T24 T28
    // T1 T5 T9 T13 T17 T21 T25 T29
    // T2 T6 T10 T14 T18 T22 T26 T30
    // T3 T7 T11 T15 T19 T23 T27 T31
    // ...
    // 每个T在K维度处理两个数据 构成8x8的块
    // 在K N维度各重复2次，构成16x16
    // Pack后 按照Tensor Core的Layout 将KxN=16x64的Block按照线程顺序在一行上排布
    // 一个T处理一个(4, (2, 4))=32xint4=128b=4xint32

    // 为了保证并行度，每次只load B Layout一列4个线程的数据
    // 换句话说，4个线程在N维上并行，每个处理(4, (2, 4))，总共处理16x64中的8列
    // T0-T7, T8-T15, T16-T23, T24-T31在K维上并行, 在原始的B矩阵中，处理的都是同一个列的数据 需要一起reduce

    // B_scale将每个Thread在处理16x64需要的8个scale放在一起
    // 8xfp16=128b
    static constexpr int kSubTileNum = 2;
    static constexpr int kElemsSubTileK = 2;
    static constexpr int kElemsPerThreadA = kSubTileNum * kElemsSubTileK; // 一个线程在K维度上处理2组2个A的数据
    static constexpr int kElemsPerColB = 4; // 详情看Tensor Core Layout
    static constexpr int kElemsPerRowB = 8; 
    static constexpr int kTileNumB = 4; // 分为4个16x16
    static constexpr int kElemsPerTileB = 8; // 每个线程在16x16中处理8个数据
    static constexpr int WARP_K = 8; // 一个warp有多少Thread在Pack后的k维度上处理
    static constexpr int WARP_N = 4; // 一个warp有多少Thread在Pack后的n维度上处理
    static constexpr int NUM_WARP = BlockSize / 32;
    static constexpr int ALL_WARP_K = NUM_WARP * WARP_K;

    const int tidx = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_idx = threadIdx.y;
    const int lane_idx = threadIdx.x;

    const int Pack_K = K / 16;
    const int Pack_N = N * 16 / 8;

    static constexpr int group_size = 128;

    // mem in reg
    // act, scale, qweight, dqweight, psum
    // TODO: BLOCK_N, BLOCK_K
    half A_reg[Batch * kElemsPerThreadA];
    int B_ori_reg[kTileNumB];
    half B_dq_reg[kElemsPerColB * kElemsPerRowB];
    half B_scale_reg[kElemsPerRowB];
    half B_scaled_zp_reg[kElemsPerRowB];

    // 这里用half还是float会影响精度，但精度损失与gemm相当
    // 精度损失主要在K处，K越大损失越大
    // half psum[Batch];
    static constexpr int ResNum = Batch * kElemsPerRowB;
    float psum[ResNum];
    #pragma unroll
    for (int i = 0; i < ResNum; i++)
        psum[i] = 0.0f;

    // mem in smem
    // out_smem(in difference warp)
    __shared__ float out_smem[NUM_WARP][ResNum];

    // Offset, Ptr and Stride
    // Pack前
    static constexpr int kSubTile = 8;
    static constexpr int kTileK = 16;
    static constexpr int kTileN = 64;
    // Pack后 注意B在N维上连续
    static constexpr int kPackTileK = 1; // = 16 / 16
    static constexpr int kPackTileN = 128; // = 64 * 16 / (int32 / int4)

    const int n_block_offset = blockIdx.x * WARP_N * kTileNumB;
    const int n_thread_offset = (lane_idx / WARP_K) * 4; // 4xint32

    const int k_offset_A = (warp_idx * WARP_K + lane_idx % WARP_K) * kTileK + (lane_idx / WARP_K) * kElemsSubTileK; // 两个half所以x2
    const int k_offset_B = (warp_idx * WARP_K * kPackTileK + lane_idx % WARP_K) * Pack_N;

    const half* A_ptr = A + k_offset_A;
    const int* B_ptr = B + n_block_offset + n_thread_offset + k_offset_B;
    const half* B_scale_ptr = B_scale + warp_idx * N + blockIdx.x * kElemsPerRowB;
    const half* B_szp_ptr = B_scaled_zp + warp_idx * N + blockIdx.x * kElemsPerRowB;

    const int k_stride_A = ALL_WARP_K * kTileK;
    const int k_stride_B = ALL_WARP_K * Pack_N;
    const int k_stride_S = NUM_WARP * N;
    
#if 0
__syncthreads();
if (tidx == 0 && blockIdx.x == 0) {
    printf("tidx: %d, n_block_offset: %d, n_thread_offset: %d, k_offset_A: %d, k_offset_B: %d\n", tidx, n_block_offset, n_thread_offset, k_offset_A, k_offset_B);
    printf("A start: %.4f\n", static_cast<float>(*(A_ptr)));
    printf("B start: %d\n", static_cast<int>(*(B_ptr)));
    printf("scale: %.4f\n", static_cast<float>(*(B_scale_ptr)));
}
__syncthreads();
if (tidx == 1 && blockIdx.x == 0) {
    printf("tidx: %d, n_block_offset: %d, n_thread_offset: %d, k_offset_A: %d, k_offset_B: %d\n", tidx, n_block_offset, n_thread_offset, k_offset_A, k_offset_B);
    printf("A start: %.4f\n", static_cast<float>(*(A_ptr)));
    printf("B start: %d\n", static_cast<int>(*(B_ptr)));
    printf("scale: %.4f\n", static_cast<float>(*(B_scale_ptr)));
}
__syncthreads();
if (tidx == 8 && blockIdx.x == 0) {
    printf("tidx: %d, n_block_offset: %d, n_thread_offset: %d, k_offset_A: %d, k_offset_B: %d\n", tidx, n_block_offset, n_thread_offset, k_offset_A, k_offset_B);
    printf("A start: %.4f\n", static_cast<float>(*(A_ptr)));
    printf("B start: %d\n", static_cast<int>(*(B_ptr)));
    printf("scale: %.4f\n", static_cast<float>(*(B_scale_ptr)));
}
__syncthreads();
if (tidx == 32 && blockIdx.x == 0) {
    printf("tidx: %d, n_block_offset: %d, n_thread_offset: %d, k_offset_A: %d, k_offset_B: %d\n", tidx, n_block_offset, n_thread_offset, k_offset_A, k_offset_B);
    printf("A start: %.4f\n", static_cast<float>(*(A_ptr)));
    printf("B start: %d\n", static_cast<int>(*(B_ptr)));
    printf("scale: %.4f\n", static_cast<float>(*(B_scale_ptr)));
}
#endif

    // main loop
    const int NUM_K = K / k_stride_A;
    EdqFastNumericArrayConverter<uint4, int, 8> convert_i4tofp16;

    for (int k = 0; k < NUM_K; k++) {
        // for (int n = 0; n < BLOCK_N; n++)

        // load B, copy 128b
        // *(reinterpret_cast<float4*>(B_ori_reg)) = *(reinterpret_cast<const float4*>(B_ptr + k * k_stride_B));
        *(reinterpret_cast<float4*>(B_ori_reg)) = __ldcg(reinterpret_cast<const float4*>(B_ptr + k * k_stride_B));

        // 16(tile) x 8(threads) == 128 这意味着每次迭代1个warp刚好处理一个group
        // load Scales and Scaled_zp
        // check group size
        *(reinterpret_cast<float4*>(B_scale_reg)) = *(reinterpret_cast<const float4*>(B_scale_ptr + k * k_stride_S));
        *(reinterpret_cast<float4*>(B_scaled_zp_reg)) = *(reinterpret_cast<const float4*>(B_szp_ptr + k * k_stride_S));
        
        // vectorize dequantization
        // 类型转换 转换后按照tile的顺序存储
        #pragma unroll
        for (int i = 0; i < kTileNumB; i++) {
            *(reinterpret_cast<uint4*>(B_dq_reg) + i) = convert_i4tofp16(B_ori_reg[i]);
        }
        // 反量化一列的数据
        #pragma unroll        
        for (int i = 0; i < kElemsPerRowB; i++) {
            half2 s = __half2half2(B_scale_reg[i]);
            half2 sz = __half2half2(B_scaled_zp_reg[i]);
            half2* B_dq_base_ptr = reinterpret_cast<half2*>(B_dq_reg + i * kElemsPerColB);
            B_dq_base_ptr[0] = __hfma2(B_dq_base_ptr[0], s, sz);
            B_dq_base_ptr[1] = __hfma2(B_dq_base_ptr[1], s, sz);
        }

        #pragma unroll
        for (int bs = 0; bs < Batch; bs++) { 
            // A * B
            // k维上的2个子分块
            #pragma unroll
            for (int i = 0; i < kSubTileNum; i++) {
                // load A
                half2* A_vec_ptr = reinterpret_cast<half2*>(A_reg + bs * kElemsPerThreadA + i * kElemsSubTileK);
                *A_vec_ptr = *(reinterpret_cast<const half2*>(A_ptr + bs * K + k * k_stride_A + i * kSubTile));
                half2* B_vec_ptr = reinterpret_cast<half2*>(B_dq_reg) + i;
                // n维上的8列
                #pragma unroll
                for(int j = 0; j < kElemsPerRowB; j++) {
                    // 现在尝试了3种实现，暂时采用fma这种，从w8a16的m=1结果来看fma大部分时候都快
                    // fma -> fma -> add
                    psum[bs * kElemsPerRowB + j] += __half2float(
                        __hfma(
                            A_vec_ptr->y,
                            (B_vec_ptr + j * kSubTileNum)->y,
                            __hfma(A_vec_ptr->x, (B_vec_ptr + j * kSubTileNum)->x, 0)
                        )
                    );
                    // mul -> mul -> add -> add
                    // psum[bs * kElemsPerRowB + j] += __half2float(A_vec_ptr->x * (B_vec_ptr + j * kSubTileNum)->x + 
                    //     A_vec_ptr->y * (B_vec_ptr + j * kSubTileNum)->y);
                    // mul2 -> add -> add
                    // half2 tmp = __hmul2(*(A_vec_ptr), *(B_vec_ptr + j * kSubTileNum));
                    // psum[bs * kElemsPerRowB + j] += __half2float(tmp.x + tmp.y);
                }
            }
        }
    }

    warp_reduce<Batch, kElemsPerRowB, 32>(psum, out_smem);

#if 0
if (tidx == 0 && blockIdx.x == 0) {
    // printf("tidx: %d, out_smem: %.4f\n", tidx, static_cast<float>(out_smem[0][0]));
    // printf("tidx: %d, out_smem: %.4f\n", tidx, static_cast<float>(out_smem[0][1]));
    // printf("tidx: %d, out_smem: %.4f\n", tidx, static_cast<float>(out_smem[0][2]));
    // printf("tidx: %d, out_smem: %.4f\n", tidx, static_cast<float>(out_smem[0][3]));
    for (int w = 0; w < NUM_WARP; w++) {
        printf("tidx: %d, block: %d, w: %d, out_smem: %.4f\n", tidx, blockIdx.x, w, static_cast<float>(out_smem[w][0]));
    }
    // for(int i = 0; i < ResNum; i++) {
    //     printf("tidx: %d, block: %d, i: %d, out_smem: %.4f\n", tidx, blockIdx.x, i, static_cast<float>(out_smem[0][i]));
    // }
}
__syncthreads();
// if (tidx == 0 && blockIdx.x == 4) {
//     for(int i = 0; i < ResNum; i++) {
//         printf("tidx: %d, block: %d, i: %d, out_smem: %.4f\n", tidx, blockIdx.x, i, static_cast<float>(out_smem[0][i]));
//     }
// }
// __syncthreads();
// if (tidx == 32 && blockIdx.x == 0) {
//     printf("tidx: %d, out_smem: %.4f\n", tidx, static_cast<float>(out_smem[0][1]));
// }
#endif

    // warp-level reduce and write back
    if (tidx < kElemsPerRowB) {
        #pragma unroll
        for (int bs = 0; bs < Batch; bs++) {
            float acc = 0.0f;
            #pragma unroll
            for (int w = 0; w < NUM_WARP; w++)
                acc += out_smem[w][bs * kElemsPerRowB + tidx];
            int n_block_offset_c = (blockIdx.x / kSubTile) * 64 + blockIdx.x % kSubTile;
            C[bs * N + n_block_offset_c + tidx * kSubTile] = __float2half(acc);
        }
    }
}


void w4a16_wa_gemv_launch(
    const half* qact,
    const int* qweight,
    half* out_feat,
    const half* w_scale,
    const half* w_scaled_zp,
    const int batch_size,
    const int out_channel,
    const int in_channel
) {
    // N维分块太大了，并行度很可能不够，需要在Batch上并行
    // 一次不要load完一个16x64的分块，改为load 4个线程即16x(1x8)
    static constexpr int NUM_WARP = 4;

    dim3 grid_dim(out_channel / 8);
    dim3 block_dim(32, NUM_WARP);

    BATCH_SWITCH(batch_size, Batch, [&] {
        auto kernel = w4a16_wa_gemv<Batch, 32 * NUM_WARP>;

        kernel<<<grid_dim, block_dim>>>(
            qact, qweight, out_feat, 
            w_scale, w_scaled_zp,
            out_channel, in_channel
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        // cudaDeviceSynchronize();
    });
}



// pybind接口
at::Tensor w4a16_wa_gemv_cuda(
    const at::Tensor &qact, // (batch_size, in_feat)
    const at::Tensor &qweight, // (in_feat // 16, out_feat * 16 // 8)
    const at::Tensor &w_scale, // (in_feat // group_size, out_feat)
    const at::Tensor &w_scaled_zp, // (in_feat // group_size, out_feat)
    std::optional<at::Tensor> &out_ // (batch_size, out_feat)
) {
    const int batch_size = qact.size(0);
    const int in_channel = qact.size(-1);
    const int out_channel = w_scale.size(-1);
    const int group_size = in_channel / w_scale.size(-2);

    TORCH_CHECK(qact.size(-1) == qweight.size(-2) * 16);
    TORCH_CHECK(qweight.size(-1) == w_scale.size(-1) * 16 / 8);
    TORCH_CHECK(w_scale.size(-1) == w_scaled_zp.size(-1));
    TORCH_CHECK(w_scale.size(-2) == w_scaled_zp.size(-2));
    TORCH_CHECK(group_size == 128);
    // printf("w4a16_wa_gemv_cuda\n");

    at::Tensor out_feat;
    if (out_.has_value()) {
        out_feat = out_.value();
        TORCH_CHECK(out_feat.dtype() == at::kHalf);
        TORCH_CHECK(out_feat.size(-2) == batch_size);
        TORCH_CHECK(out_feat.size(-1) == out_channel);
        TORCH_CHECK(out_feat.is_cuda(), "Out_feat must be on CUDA");
    }
    else {
        out_feat = torch::empty({batch_size, out_channel}, qact.options().dtype(at::kHalf));
    }

    auto qact_ptr = reinterpret_cast<half*>(qact.data_ptr());
    auto qweight_ptr = reinterpret_cast<int*>(qweight.data_ptr());
    auto w_scale_ptr = reinterpret_cast<half*>(w_scale.data_ptr());
    auto w_scaled_zp_ptr = reinterpret_cast<half*>(w_scaled_zp.data_ptr());
    auto out_feat_ptr = reinterpret_cast<half*>(out_feat.data_ptr());

    w4a16_wa_gemv_launch(
        qact_ptr, qweight_ptr, out_feat_ptr, 
        w_scale_ptr, w_scaled_zp_ptr,
        batch_size, out_channel, in_channel
    );


    return out_feat;
}