#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>

__device__ __forceinline__ void permute_2_uint4(const uint4 a, const uint4 b, uint4& c, uint4& d) {
    const uint mask_l = 0x0f0f0f0f;
    const uint mask_h = 0xf0f0f0f0;
    
    c.x = ((b.x & mask_l) << 4) | (a.x & mask_l);
    c.y = ((b.y & mask_l) << 4) | (a.y & mask_l);
    c.z = ((b.z & mask_l) << 4) | (a.z & mask_l);
    c.w = ((b.w & mask_l) << 4) | (a.w & mask_l);
    
    d.x = ((a.x & mask_h) >> 4) | (b.x & mask_h);
    d.y = ((a.y & mask_h) >> 4) | (b.y & mask_h);
    d.z = ((a.z & mask_h) >> 4) | (b.z & mask_h);
    d.w = ((a.w & mask_h) >> 4) | (b.w & mask_h);
}

__global__ void trans_layout_c16_to_c8_kernel(
        uint4* __restrict__ output,
        const uint4* __restrict__ input,
        const int R, const int C) {
    // every block will solve the layout transformation for 16 * int32(8 * 4bit) = 4 * uint4 each turn.
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int elems_per_thread = 4; 
    const int solve_end = C / elems_per_thread;

    uint4 reg_input[4];  
    uint4 reg_output[4];  
    const uint4* input_ptr = input + row * C;
    uint4* output_ptr = output + row * C;
    for (int cur_tid = tid; cur_tid < solve_end; cur_tid += blockDim.x) {
        const int k_idx = cur_tid * elems_per_thread;

        reg_input[0] = __ldcg(input_ptr + 0 + k_idx);
        reg_input[1] = __ldcg(input_ptr + 1 + k_idx);
        reg_input[2] = __ldcg(input_ptr + 2 + k_idx);
        reg_input[3] = __ldcg(input_ptr + 3 + k_idx);

        // Permute 2*2*4 int32 data
        permute_2_uint4(reg_input[0], reg_input[1], reg_output[0], reg_output[2]);
        permute_2_uint4(reg_input[2], reg_input[3], reg_output[1], reg_output[3]);

        // Write-through to global memory.
        __stwt(output_ptr + 0 + k_idx, reg_output[0]);
        __stwt(output_ptr + 1 + k_idx, reg_output[1]);
        __stwt(output_ptr + 2 + k_idx, reg_output[2]);
        __stwt(output_ptr + 3 + k_idx, reg_output[3]);
    }
}

void trans_layout_c16_to_c8_marlin(
        torch::Tensor &output,      // [K/16, N*16/8], dtype=torch.int32
        torch::Tensor &input        // [K/16, N*16/8], dtype=torch.int32
    ) {
    const int R = input.size(0);
    const int C = input.size(1);
    
    const dim3 grid(R);
    const dim3 block(128);
    
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    trans_layout_c16_to_c8_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<uint4*>(output.data_ptr()),
        reinterpret_cast<const uint4*>(input.data_ptr()),
        R, C/4);
}


// The layout _permute for W4A8 is: (Output)
// tensor([[   0,   32,    8,   40,   16,   48,   24,   56],
//         [ 256,  288,  264,  296,  272,  304,  280,  312],
//         [ 512,  544,  520,  552,  528,  560,  536,  568],
//         [ 768,  800,  776,  808,  784,  816,  792,  824],
//         [  64,   96,   72,  104,   80,  112,   88,  120],
//         [ 320,  352,  328,  360,  336,  368,  344,  376],
//         [ 576,  608,  584,  616,  592,  624,  600,  632],
//         [ 832,  864,  840,  872,  848,  880,  856,  888],
//         [ 128,  160,  136,  168,  144,  176,  152,  184],
//         [ 384,  416,  392,  424,  400,  432,  408,  440],
//         [ 640,  672,  648,  680,  656,  688,  664,  696],
//         [ 896,  928,  904,  936,  912,  944,  920,  952],
//         [ 192,  224,  200,  232,  208,  240,  216,  248],
//         [ 448,  480,  456,  488,  464,  496,  472,  504],
//         [ 704,  736,  712,  744,  720,  752,  728,  760],
//         [ 960,  992,  968, 1000,  976, 1008,  984, 1016]]) ... of shape(32,32)
// The layout _permute for W4A16 is: (Input)
// tensor([[   0,  128,    8,  136,   16,  144,   24,  152],
//         [ 256,  384,  264,  392,  272,  400,  280,  408],
//         [ 512,  640,  520,  648,  528,  656,  536,  664],
//         [ 768,  896,  776,  904,  784,  912,  792,  920],
//         [  32,  160,   40,  168,   48,  176,   56,  184],
//         [ 288,  416,  296,  424,  304,  432,  312,  440],
//         [ 544,  672,  552,  680,  560,  688,  568,  696],
//         [ 800,  928,  808,  936,  816,  944,  824,  952],
//         [  64,  192,   72,  200,   80,  208,   88,  216],
//         [ 320,  448,  328,  456,  336,  464,  344,  472],
//         [ 576,  704,  584,  712,  592,  720,  600,  728],
//         [ 832,  960,  840,  968,  848,  976,  856,  984],
//         [  96,  224,  104,  232,  112,  240,  120,  248],
//         [ 352,  480,  360,  488,  368,  496,  376,  504],
//         [ 608,  736,  616,  744,  624,  752,  632,  760],
//         [ 864,  992,  872, 1000,  880, 1008,  888, 1016]]) ... of shape(32,32)
// Every row (8 * int4) store in int32 
// Every 4 row store in uint4, and they are grouped, the high and low bits can be permute together.

__global__ void __launch_bounds__(256, 2) trans_layout_kernel(
    const uint16_t* __restrict__ awq_packed,   // (N/4, K)  int16
    int8_t*  __restrict__ qserve_packed,        // (N, K/2)  int8  <-- 修正: uint8_t -> int8_t
    int N, int K
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * (K >> 1);
    if (__builtin_expect(idx >= total, 0)) return;

    const int K32 = K >> 5;
    /* ---- QServe 输出位置分解 ---- */
    int tmp = idx;
    int k3_q = tmp & 3;       tmp >>= 2;
    int n2_q = tmp & 1;       tmp >>= 1;
    int k1_q = tmp & 1;       tmp >>= 1;
    int k2_q = tmp & 3;       tmp >>= 2;
    int n3_q = tmp & 7;       tmp >>= 3;
    int k0_q = tmp % K32;     tmp  /= K32;
    int n0_q = tmp;

    const int n_base = (n0_q << 5) | (n2_q << 3) | n3_q;
    const int k_base = (k0_q << 5) | (k1_q << 4) | (k2_q << 2) | k3_q;

    /* ---- AWQ k 分量公共部分 ---- */
    const int k_val = k_base;
    const int k0_awq  = k_val >> 6;
    const int k1_awq  = (k_val >> 5) & 1;
    const int k2_awq  = (k_val >> 3) & 3;
    const int k3_awq  = (k_val >> 1) & 3;
    const int k4_awq  = k_val & 1;
    const int k_flat_common = (k0_awq << 6) | (k1_awq << 3) | (k3_awq << 1) | k4_awq;

    uint32_t result = 0;

    /* ---- n1_q = 0 (低位 nibble) ---- */
    {
        const int n = n_base;
        const int n0 = n >> 2;
        const int n1_awq = n & 3;
        const int k_flat = k_flat_common | (n1_awq << 4);
        const uint16_t val16 = __ldg(awq_packed + n0 * K + k_flat);
        const int shift = k2_awq << 2;

        uint32_t nibble;
        asm volatile ("bfe.u32 %0, %1, %2, 4;"
                      : "=r"(nibble)
                      : "r"((uint32_t)val16), "r"(shift));
        result = nibble;
    }
    /* ---- n1_q = 1 (高位 nibble) ---- */
    {
        const int n = n_base | 16;          // n1_q = 1
        const int n0 = n >> 2;
        const int n1_awq = n & 3;
        const int k_flat = k_flat_common | (n1_awq << 4);
        const uint16_t val16 = __ldg(awq_packed + n0 * K + k_flat);
        const int shift = k2_awq << 2;

        uint32_t nibble;
        asm volatile ("bfe.u32 %0, %1, %2, 4;"
                      : "=r"(nibble)
                      : "r"((uint32_t)val16), "r"(shift));

        asm volatile ("bfi.b32 %0, %1, %2, 4, 4;"
                      : "=r"(result)
                      : "r"(nibble), "r"(result));
    }
    qserve_packed[idx] = static_cast<int8_t>(result);
}

torch::Tensor trans_layout_c16_to_c8_awq(torch::Tensor awq_packed, int N, int K) {
    TORCH_CHECK(awq_packed.is_cuda(), "awq_packed must be a CUDA tensor");
    TORCH_CHECK(awq_packed.dtype() == torch::kInt16,
                "awq_packed must be int16, got ", awq_packed.dtype());
    TORCH_CHECK(N % 32 == 0 && K % 64 == 0,
                "N must be divisible by 32 and K by 64");

    auto options = torch::TensorOptions().dtype(torch::kInt8).device(awq_packed.device());
    auto qserve_packed = torch::empty({N, K / 2}, options);

    const int total   = N * (K >> 1);
    const int threads = 256;
    const int blocks  = (total + threads - 1) / threads;

    trans_layout_kernel<<<blocks, threads>>>(
        reinterpret_cast<const uint16_t*>(awq_packed.data_ptr<int16_t>()),
        // awq_packed.data_ptr<uint16_t>(),
        qserve_packed.data_ptr<int8_t>(),   // <-- 修正: uint8_t -> int8_t
        N, K
    );
    return qserve_packed;
}