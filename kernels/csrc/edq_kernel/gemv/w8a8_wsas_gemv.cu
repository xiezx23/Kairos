// Inspired by AWQ.
// Author: Junwei Chen.

#include <cuda_fp16.h>

// #include <cute/tensor.hpp>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

#include "w8a8_wsas_gemv.h"
#include "utils.h"

// Reduce sum within the warp
template <int Batch, int CTA_N, int Thread>
__forceinline__ __device__ static void warp_reduce(int* psum, int (*out_smem)[Batch]) {
    int sum_i32[Batch];
    #pragma unroll
    for (int i = 0; i < Batch; i++)
        // sum_i32[i] = static_cast<float>(psum[i]);
        sum_i32[i] = psum[i];

    // warp内reduce
    #pragma unroll
    for (int i = 0; i < Batch; i++) {
        for (int thr = (Thread >> 1); thr > 0; thr = (thr >> 1)) {
            sum_i32[i] += __shfl_xor_sync(0xffffffff, sum_i32[i], thr);
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
            out_smem[warp_idx * CTA_N + lane_idx / WARP_K][i] = sum_i32[i];

    }
    
    __syncthreads();
}


template <int CTA_N, int Batch, int BlockSize>
__global__ void w8a8_wsas_gemv(
    int8_t* A,
    int8_t* B,
    half* C,
    half* A_scale,
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
    // qact, qweight, psum
    int8_t A_reg[kElemsPerThread];
    int8_t B_reg[kElemsPerThread];

    int psum[Batch];
    for (int i = 0; i < Batch; i++)
        psum[i] = 0;

    // mem in smem
    __shared__ int out_smem[NUM_WARP * CTA_N][Batch];

    // Offset, Ptr and Stride
    const int k_offset = (warp_idx * WARP_K + lane_idx % WARP_K) * kElemsPerThread;
    const int n_block_offset = blockIdx.x * CTA_N;
    const int n_thread_offset = lane_idx / WARP_K;
    
    // Thread Layout in B(if CTA_N == 4):
    // T0 T8 T16 T24
    // T1 T9 T17 T25
    // ...
    // T沿着K维处理128b

    int8_t* A_ptr = A + k_offset;
    int8_t* B_ptr = B + (n_block_offset + n_thread_offset) * K + k_offset;
    
    // half* B_scale_ptr = B_scale + n_block_offset + n_thread_offset;

    const int k_stride = ALL_WARP_K * kElemsPerThread;

    // main loop
    // int8 x int8 -> dequantize
    const int kBound = K / k_stride;

    for (int k = 0; k < kBound; k++) {
        // G2R B, 一次load 16 x int8, 共128b
        *(reinterpret_cast<float4*>(B_reg)) = *(reinterpret_cast<float4*>(B_ptr + k * k_stride));

        // Compute
        #pragma unroll
        for (int bs = 0; bs < Batch; bs++) {
            // G2R A, 一次load 16 x int8, 共128b
            int8_t* A_load_ptr = A_ptr + bs * K + k * k_stride;
            *(reinterpret_cast<float4*>(A_reg)) = *(reinterpret_cast<float4*>(A_load_ptr));
            
            // int8 vector x int8 vector
            int* A_base_ptr = reinterpret_cast<int*>(A_reg);
            int* B_base_ptr = reinterpret_cast<int*>(B_reg);

#if 0
__syncthreads();
if (tidx == 0 && blockIdx.x == 0) {
    printf("A reg 0: %d\n", static_cast<int>(A_reg[0]));
    printf("A reg 1: %d\n", static_cast<int>(A_reg[1]));
    printf("A reg 2: %d\n", static_cast<int>(A_reg[2]));
    printf("A reg 3: %d\n", static_cast<int>(A_reg[3]));
    printf("B reg 0: %d\n", static_cast<int>(B_reg[0]));
    printf("B reg 1: %d\n", static_cast<int>(B_reg[1]));
    printf("B reg 2: %d\n", static_cast<int>(B_reg[2]));
    printf("B reg 3: %d\n", static_cast<int>(B_reg[3]));
}
#endif
            // 这里对psum的依赖会严重影响性能
            int acc = 0;
            #pragma unroll
            for (int i = 0; i < 128 / 32; i++) {
                acc = __dp4a(A_base_ptr[i], B_base_ptr[i], acc);
            }
            psum[bs] += acc;
        }
    }

#if 0
__syncthreads();
if (tidx == 0 && blockIdx.x == 0) {
    printf("psum 0: %d\n", psum[0]);
}
#endif

    // Warp Reduce(splik-k结果)
    warp_reduce<Batch, CTA_N, WARP_K>(psum, out_smem);

    // warp之间通过Smem进行Reduce
    // 进行反量化(获取act_scale和w_scale)
    // 最后进行写回Gmem
    if (warp_idx == 0 && lane_idx % WARP_K == 0) {
        float b_scale_reg = __half2float(*(B_scale + n_block_offset + n_thread_offset));
        #pragma unroll
        for (int bs = 0; bs < Batch; bs++) {
            int acc = 0;
            #pragma unroll
            for (int w = 0; w < NUM_WARP; w++)
                acc += out_smem[w * CTA_N + lane_idx / WARP_K][bs];
            // Dequantization
            float acc_fp32 = __int2float_rn(acc);
            float a_scale_reg = __half2float(*(A_scale + bs));
            half acc_fp16 = __float2half(acc_fp32 * a_scale_reg * b_scale_reg);
            C[bs * N + n_block_offset + n_thread_offset] = acc_fp16;
        }
    }
}


void w8a8_wsas_gemv_launch(
    int8_t* qact,
    int8_t* qweight,
    half* out_feat,
    half* act_scale,
    half* w_scale,
    const int batch_size,
    const int out_channel,
    const int in_channel
) {

    static constexpr int CTA_N = 8; // 1/2/4/8/16/32->K Align=2048/1024/512/256/128/64
    static constexpr int NUM_WARP = 4;

    dim3 grid_dim(out_channel / CTA_N);
    dim3 block_dim(32, NUM_WARP);

    BATCH_SWITCH(batch_size, Batch, [&] {
        auto kernel = w8a8_wsas_gemv<CTA_N, Batch, 32 * NUM_WARP>;

        kernel<<<grid_dim, block_dim>>>(
            qact, qweight, out_feat, 
            act_scale, w_scale,
            out_channel, in_channel
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        // cudaDeviceSynchronize();
    });
}


// pybind接口
at::Tensor w8a8_wsas_gemv_cuda(
    const at::Tensor &qact, // (batch_size, in_feat)
    const at::Tensor &qweight, // (out_feat, in_feat)
    const at::Tensor &act_scale, // (batch_size)
    const at::Tensor &w_scale, // (out_feat)
    std::optional<at::Tensor> &out_ // (batch_size, out_feat)
) {
    const int batch_size = qact.size(0);
    const int in_channel = qact.size(-1);
    const int out_channel = qweight.size(-2);
    TORCH_CHECK(qact.size(-1) == qweight.size(-1));
    TORCH_CHECK(qact.size(-2) == act_scale.size(-1));
    TORCH_CHECK(qweight.size(-2) == w_scale.size(-1));
    // printf("w8a8_wsas_gemv_cuda\n");

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

    auto qact_ptr = reinterpret_cast<int8_t*>(qact.data_ptr<int8_t>());
    auto qweight_ptr = reinterpret_cast<int8_t*>(qweight.data_ptr<int8_t>());
    auto act_scale_ptr = reinterpret_cast<half*>(act_scale.data_ptr());
    auto w_scale_ptr = reinterpret_cast<half*>(w_scale.data_ptr());
    auto out_feat_ptr = reinterpret_cast<half*>(out_feat.data_ptr());

    w8a8_wsas_gemv_launch(
        qact_ptr, qweight_ptr, out_feat_ptr, 
        act_scale_ptr, w_scale_ptr,
        batch_size, out_channel, in_channel
    );
    
    return out_feat;
}