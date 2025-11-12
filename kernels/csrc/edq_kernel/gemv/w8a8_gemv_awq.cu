// Inspired by AWQ.
// Author: Junwei Chen.
// NOTE: No used.

template <int Num, int Thread>
__forceinline__ __device__ static void warp_reduce(int* psum, int (*out_smem)[Num]) {
    int sum_i32[Num];
    #pragma unroll
    for (int i = 0; i < Num; i++)
        sum_i32[i] = psum[i];

    // warp内reduce
    #pragma unroll
    for (int i = 0; i < Num; i++) {
        for (int thr = (Thread >> 1); thr > 0; thr = (thr >> 1)) {
            sum_i32[i] += __shfl_xor_sync(0xffffffff, sum_i32[i], thr);
        }
    }

    __syncthreads();

    // R2S
    int lane_idx = threadIdx.x;
    int warp_idx = threadIdx.y;
    // maybe WARP_K 
    if (lane_idx == 0) {
        #pragma unroll
        for (int i = 0; i < Num; i++)
            out_smem[warp_idx][i] = sum_i32[i];

    }
    
    __syncthreads();
}


// 参考AWQ并行方式的实现，所有warp都在K上并行
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
    static constexpr int NUM_WARP = BlockSize / 32;

    const int tidx = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_idx = threadIdx.y;
    const int lane_idx = threadIdx.x;

    const int Num = Batch * CTA_N;

    // mem in reg
    // qact, qweight, psum
    int8_t A_reg[kElemsPerThread];
    int8_t B_reg[kElemsPerThread * CTA_N];

    int psum[Num];
    for (int i = 0; i < Num; i++)
        psum[i] = 0;

    // mem in smem
    __shared__ int out_smem[NUM_WARP][Batch * CTA_N];

    // Offset, Ptr and Stride
    const int n_block_offset = blockIdx.x * CTA_N;

    int8_t* B_ptr = B + (n_block_offset) * K;

    constexpr int k_stride = BlockSize * kElemsPerThread;

    // main loop
    // int8 x int8 -> dequantize
    for (int k = tidx * kElemsPerThread; k < K; k += k_stride) {
        #pragma unroll
        for (int n = 0; n < CTA_N; n++) {
            // G2R B, 一次load 16 x int8, 共128b
            *(reinterpret_cast<float4*>(B_reg + n * kElemsPerThread)) =
                *(reinterpret_cast<float4*>(B_ptr + n * K + k));
            int* B_base_ptr = reinterpret_cast<int*>(B_reg + n * kElemsPerThread);

            // Compute
            #pragma unroll
            for (int bs = 0; bs < Batch; bs++) {
                // G2R A, 一次load 16 x int8, 共128b
                int8_t* A_ptr = A + bs * K + k;
                *(reinterpret_cast<float4*>(A_reg)) = *(reinterpret_cast<float4*>(A_ptr));
                
                // int8 vector x int8 vector
                int* A_base_ptr = reinterpret_cast<int*>(A_reg);

                // 这里对psum的依赖会严重影响性能
                int acc = 0;
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    acc = __dp4a(A_base_ptr[i], B_base_ptr[i], acc);
                }
                psum[bs * CTA_N + n] += acc;
            }
        }
    }

#if 0
__syncthreads();
if (tidx == 0 && blockIdx.x == 0) {
    printf("psum 0: %d\n", psum[0]);
}
#endif

    // Warp Reduce(splik-k结果)
    warp_reduce<Num, 32>(psum, out_smem);

    // warp之间通过Smem进行Reduce
    // 进行反量化(获取act_scale和w_scale)
    // 最后进行写回Gmem
    for (int i = tidx; i < Num; i+= BlockSize) {
        int bs = i / CTA_N;
        int n = i % CTA_N;
        
        float a_scale_reg = __half2float(*(A_scale + bs));
        float b_scale_reg = __half2float(*(B_scale + n_block_offset + n));

        int acc = 0;
        #pragma unroll
        for (int w = 0; w < NUM_WARP; w++)
            acc += out_smem[w][bs * CTA_N + n];
        // Dequantization
        float acc_fp32 = __int2float_rn(acc);
        half acc_fp16 = __float2half(acc_fp32 * a_scale_reg * b_scale_reg);
        C[bs * N + n_block_offset + n] = acc_fp16;
    }
}
