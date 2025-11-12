// Inspired by AWQ and cute gemm written by reed.
// Author: Junwei Chen.

#include <cuda_fp16.h>

#include <cute/tensor.hpp>
#include <cute/util/debug.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

#include "w8a16_ws_gemm.h"

#include "utils.h"

using namespace cute;


template <
    int CTA_M_, int CTA_N_, int CTA_K_,
    int WARP_M_, int WARP_N_, int WARP_K_,
    int STAGES_, bool IS_EVEN_M_,
    typename DTypeA_, typename DTypeB_, typename DTypeC_
>
struct KernelTraits {
    // 对于Type，目前考虑DQ后fp16 GEMM，反量化因子的类型是fp16
    // DTypeA=fp16, DTypeB=int8

    // ABC在输入输出时的类型
    using DTypeA = DTypeA_;
    using DTypeB = DTypeB_;
    using DTypeC = DTypeC_;
 
    using Element = cutlass::half_t; // GEMM时的Type
    using ElementAccum = float; // 累加的Type
    
    using ScaleType = cutlass::half_t;

    // 分组量化每组元素数量
    // static constexpr int GROUP_SIZE = CTA_K_;

    // 分块大小
    static constexpr int CTA_M = CTA_M_;
    static constexpr int CTA_N = CTA_N_;
    static constexpr int CTA_K = CTA_K_;

    // 每个Warp负责计算的分块大小
    static constexpr int WARP_SIZE = 32;
    static constexpr int WARP_M = WARP_M_;
    static constexpr int WARP_N = WARP_N_;
    static constexpr int WARP_K = WARP_K_;
    // 根据CTA分块大小和每个Warp负责计算的分块大小，确定使用多少个Warp
    static constexpr int NUM_WARP_M = (CTA_M / WARP_M);
    static constexpr int NUM_WARP_N = (CTA_N / WARP_N);
    static constexpr int NUM_WARP_K = (CTA_K / WARP_K);
    static constexpr int NUM_WARPS_MN = NUM_WARP_M * NUM_WARP_N;
    static constexpr int NUM_WARPS = NUM_WARP_M * NUM_WARP_N * NUM_WARP_K;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_SIZE;

    //TODO: 暂时不实现split-K
    static constexpr bool SPLIT_K = NUM_WARP_K > 1;
    static constexpr int STAGES = STAGES_;

    static constexpr bool IS_EVEN_M = IS_EVEN_M_;

    // MMA
    using MMA_OP = std::conditional_t<
        std::is_same_v<ElementAccum, float>,
        SM80_16x8x16_F32F16F16F32_TN,
        SM80_16x8x16_F16F16F16F16_TN
    >;
    using MMA_TRAIT = MMA_Traits<MMA_OP>;
    using MMA_ATOM = MMA_Atom<MMA_TRAIT>;
    using MMA_ATOM_SHAPE = MMA_TRAIT::Shape_MNK;

    using MMAThrLayout = decltype(make_layout(make_shape(
        Int<NUM_WARP_M>{}, Int<NUM_WARP_N>{}, Int<NUM_WARP_K>{})));

    static constexpr int kMmaPermuteM = NUM_WARP_M * get<0>(MMA_ATOM_SHAPE{});
    static constexpr int kMmaPermuteN = NUM_WARP_N * get<1>(MMA_ATOM_SHAPE{}) * 2;
    static constexpr int kMmaPermuteK = NUM_WARP_K * get<2>(MMA_ATOM_SHAPE{});

    using kMmaPermuteNLayout =
        Layout<Shape<_2, _4, Int<NUM_WARP_N>, _2>, Stride<_1, _2, _16, _8>>;

    using MMAPermutation = decltype(make_tile(
        Int<kMmaPermuteM>{}, kMmaPermuteNLayout{}, Int<kMmaPermuteK>{}));

    using TiledMma = TiledMMA<
        MMA_ATOM,
        MMAThrLayout,
        MMAPermutation
    >;

    // swizzle
    // static constexpr int kShmLoadSwizzleB = 3; // 2^3=8 row
    // static constexpr int kShmLoadSwizzleM = 3; // 2^3=8 col = 8xfp16 = 16Byte = uint128
    // static constexpr int kShmLoadSwizzleS = cutlass::log2_down<CTA_K / (1 << kShmLoadSwizzleM)>::value; // S=3 -> 2^3组M一行

    // 这里不一定要限定在列上的长度是CTA_K，可以设置为64之类的，和CTA_K解耦，支持更多的CTA_K设置
    // 不过对于int8，一行还得是128个，或者说能不能一次load多行，对多行进行swizzle
    using SmemLayoutAtomA = decltype(
        composition(
            Swizzle<3, 3, 3>{}, // 2^3=8 row, 2^3=8 ele/group, 2^3=8 group/col
            Layout<
                Shape<_8, _64>,
                Stride<_64, _1>
            >{}
        )
    );

    // int 8
    using SmemLayoutAtomB = decltype(
        composition(
            Swizzle<3, 4, 3>{}, // 2^3=8 row, 2^4=16 ele/group, 2^3=8 group/col
            Layout<
                Shape<_8, _64>,
                Stride<_64, _1>
            >{}
        )
    );

    using ShapeA = Shape<Int<CTA_M>, Int<CTA_K>, Int<STAGES>>;
    using ShapeB = Shape<Int<CTA_N>, Int<CTA_K>, Int<STAGES>>;
    using ShapeC = Shape<Int<CTA_M>, Int<CTA_N>>;

    using SmemLayoutA = decltype(
        tile_to_shape(
            SmemLayoutAtomA{},
            ShapeA{}
        )
    );

    using SmemLayoutB = decltype(
        tile_to_shape(
            SmemLayoutAtomB{},
            ShapeB{}
        )
    );

    using SmemLayoutS = Layout<
        Shape<Int<CTA_N>>,
        Stride<_1>
    >;

    using SmemLayoutAtomC = decltype(
        composition(
            Swizzle<3, 3, 3>{},
            Layout<
                Shape<_8, _64>,
                Stride<_64, _1>
            >{}
        )
    );

    using SmemLayoutCReduce = decltype(tile_to_shape(
        SmemLayoutAtomC{},
        Shape<Int<CTA_M>, Int<CTA_N>, Int<NUM_WARP_K>>{})
    );

    static constexpr int kSmemSizeA = size(SmemLayoutA{}) * sizeof(DTypeA);
    static constexpr int kSmemSizeB = size(SmemLayoutB{}) * sizeof(DTypeB);
    static constexpr int kSmemSizeS = size(SmemLayoutS{}) * sizeof(ScaleType);
    static constexpr int kSmemSizeCReduce = size(SmemLayoutCReduce{}) * (SPLIT_K ? sizeof(ElementAccum): sizeof(DTypeC));
    static constexpr int kSmemSize = 
        (kSmemSizeA + kSmemSizeB + kSmemSizeS) > kSmemSizeCReduce 
        ? (kSmemSizeA + kSmemSizeB + kSmemSizeS) 
        : kSmemSizeCReduce;

    // Copy fp16 A, int8 B G2S
    static constexpr int kGmemElemsPerLoadA = sizeof(cute::uint128_t) / sizeof(DTypeA); // 8
    static constexpr int kGmemElemsPerLoadB = sizeof(cute::uint128_t) / sizeof(DTypeB); // 16
    static constexpr int kGmemElemsPerLoadS = sizeof(cutlass::half_t) / sizeof(ScaleType);  // 1

    // 每行有多少个线程用于load
    static constexpr int kGmemThreadsPerRowA = CTA_K / kGmemElemsPerLoadA;
    static constexpr int kGmemThreadsPerRowB = CTA_K / kGmemElemsPerLoadB;
    static constexpr int kGmemThreadsPerRowS = CTA_N / kGmemElemsPerLoadS;
    
    // 每列有多少线程进行load
    static constexpr int kGmemThreadsPerColA = NUM_THREADS / kGmemThreadsPerRowA;
    static constexpr int kGmemThreadsPerColB = NUM_THREADS / kGmemThreadsPerRowB;
    static constexpr int kGmemThreadsPerColS = NUM_THREADS / kGmemThreadsPerRowS;

    using G2SLayoutAtomA = Layout<
        Shape<Int<kGmemThreadsPerColA>, Int<kGmemThreadsPerRowA>>,
        Stride<Int<kGmemThreadsPerRowA>, _1>
    >;

    using G2SLayoutAtomB = Layout<
        Shape<Int<kGmemThreadsPerColB>, Int<kGmemThreadsPerRowB>>,
        Stride<Int<kGmemThreadsPerRowB>, _1>
    >;

    using G2SLayoutAtomS = Layout<
        Shape<Int<kGmemThreadsPerRowS>>,
        Stride<_1>
    >;

    using CopyAtomG2SA = std::conditional_t<
        STAGES == 1,
        Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, DTypeA>,
        Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, DTypeA>
    >;

    using CopyAtomG2SB = std::conditional_t<
        STAGES == 1,
        Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, DTypeB>,
        Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, DTypeB>
    >;

    using CopyAtomG2SS = Copy_Atom<DefaultCopy, ScaleType>;

    using TiledCopyG2SA = decltype(
        make_tiled_copy(
            CopyAtomG2SA{},
            G2SLayoutAtomA{},
            Layout<Shape<_1, Int<kGmemElemsPerLoadA>>>{}
        )
    );

    using TiledCopyG2SB = decltype(
        make_tiled_copy(
            CopyAtomG2SB{},
            G2SLayoutAtomB{},
            Layout<Shape<_1, Int<kGmemElemsPerLoadB>>>{}
        )
    );

    using TiledCopyG2SS = decltype(
        make_tiled_copy(
            CopyAtomG2SS{},
            G2SLayoutAtomS{},
            Layout<Shape<Int<kGmemElemsPerLoadS>>>{}
        )
    );

    // Copy fp16 A, int8 B S2R
    using CopyAtomS2RA = Copy_Atom<SM75_U32x4_LDSM_N, DTypeA>;

    using CopyAtomS2RB = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<16>, DTypeB>;

    // Copy fp16 C R2S, 在copy前完成fp32到fp16的转换
    using CopyAtomR2SC = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<32>, DTypeC>;

    // Copy fp16 C S2R2G
    using CopyAtomS2GC = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, DTypeC>;
    static constexpr int kGmemElemsPerStore = sizeof(cute::uint128_t) / sizeof(DTypeC); // 8
    // 多少个线程用于store一行
    static constexpr int kGmemThreadsPerRowC = CTA_N / kGmemElemsPerStore; 
    // 每列有多少线程进行store
    static constexpr int kGmemThreadsPerColC = NUM_THREADS / kGmemThreadsPerRowC;

    using S2GLayoutAtom = Layout<
        Shape<Int<kGmemThreadsPerColC>, Int<kGmemThreadsPerRowC>>,
        Stride<Int<kGmemThreadsPerRowC>, _1>
    >;

    using TiledCopyS2GC = decltype(
        make_tiled_copy(
            CopyAtomS2GC{},
            S2GLayoutAtom{},
            Layout<Shape<_1, Int<kGmemElemsPerStore>>>{}
        )
    );

};


template <typename KernelTraits>
__global__ void w8a16_ws_gemm(
    cutlass::half_t* __restrict__ A,
    typename KernelTraits::DTypeB* __restrict__ B,
    cutlass::half_t* __restrict__ C,
    cutlass::half_t* __restrict__ B_scale,
    const int M,
    const int N,
    const int K
) {
    // KernelTraits
    using DTypeA = typename KernelTraits::DTypeA;
    using DTypeB = typename KernelTraits::DTypeB;
    using DTypeC = typename KernelTraits::DTypeC;
    using Element = typename KernelTraits::Element;
    using ElementAccum = typename KernelTraits::ElementAccum;
    using ScaleType = typename KernelTraits::ScaleType;

    static constexpr int CTA_M = KernelTraits::CTA_M;
    static constexpr int CTA_N = KernelTraits::CTA_N;
    static constexpr int CTA_K = KernelTraits::CTA_K;
    static constexpr int WARP_M = KernelTraits::WARP_M;
    static constexpr int WARP_N = KernelTraits::WARP_N;
    static constexpr int WARP_K = KernelTraits::WARP_K;
    static constexpr int NUM_WARP_K = KernelTraits::NUM_WARP_K;
    static constexpr int NUM_WARPS = KernelTraits::NUM_WARPS;
    static constexpr int NUM_THREADS = KernelTraits::NUM_THREADS;
    static constexpr bool SPLIT_K = KernelTraits::SPLIT_K;
    static constexpr int STAGES = KernelTraits::STAGES;

    const int num_blocks_m = (M + CTA_M - 1) / CTA_M;
    const int num_blocks_n = (N + CTA_N - 1) / CTA_N;

    const int log_tile = get_log_tile<8>(num_blocks_m);
    const uint2 block_idx_mapping = get_block_idx_mapping(blockIdx.x, blockIdx.y, log_tile);
    const int n_block_idx = block_idx_mapping.x;
    const int m_block_idx = block_idx_mapping.y;

    // 防止多余的block参与计算
    if (m_block_idx >= num_blocks_m || n_block_idx >= num_blocks_n)
        return;

    // 是否需要边界检查
    const bool NeedCheck = m_block_idx == num_blocks_m - 1 && !KernelTraits::IS_EVEN_M;

    const int lane_idx = threadIdx.x;
    const int warp_idx = threadIdx.y;
    const int warp_mn_idx = warp_idx % KernelTraits::NUM_WARPS_MN;
    const int warp_k_idx = warp_idx / KernelTraits::NUM_WARPS_MN;
    const int tidx = threadIdx.x + threadIdx.y * blockDim.x;

    extern __shared__ char smem_[];

    // gmem tensor
    Tensor mA = make_tensor(
        make_gmem_ptr(reinterpret_cast<DTypeA*>(A)),
        make_shape(M, K),
        make_stride(K, Int<1>{})
    );
    
    Tensor mB = make_tensor(
        make_gmem_ptr(reinterpret_cast<DTypeB*>(B)),
        make_shape(N, K),
        make_stride(K, Int<1>{})
    );
    
    Tensor mS = make_tensor(
        make_gmem_ptr(reinterpret_cast<ScaleType*>(B_scale)),
        make_shape(N),
        make_stride(Int<1>{})
    );

    Tensor mC = make_tensor(
        make_gmem_ptr(reinterpret_cast<DTypeC*>(C)),
        make_shape(M, N),
        make_stride(N, Int<1>{})
    );

    Tensor mA_pred = make_identity_tensor(shape(mA));
    Tensor mC_pred = make_identity_tensor(shape(mC));

    // gmem tensor tile
    Tensor gA = local_tile(
        mA,
        Shape<Int<CTA_M>, Int<CTA_K>>{},
        make_coord(m_block_idx, _)
    );

    Tensor gB = local_tile(
        mB,
        Shape<Int<CTA_N>, Int<CTA_K>>{},
        make_coord(n_block_idx, _)
    );

    Tensor gS = local_tile(
        mS,
        Shape<Int<CTA_N>>{},
        make_coord(n_block_idx)
    );

    Tensor gC = local_tile(
        mC,
        Shape<Int<CTA_M>, Int<CTA_N>>{},
        make_coord(m_block_idx, n_block_idx)
    );

    Tensor gA_pred = local_tile(
        mA_pred,
        Shape<Int<CTA_M>, Int<CTA_K>>{},
        make_coord(m_block_idx, _)
    );

    Tensor gC_pred = local_tile(
        mC_pred,
        Shape<Int<CTA_M>, Int<CTA_N>>{},
        make_coord(m_block_idx, n_block_idx)
    );

    // smem tensor
    Tensor sA = make_tensor(
        make_smem_ptr(reinterpret_cast<DTypeA*>(smem_)),
        typename KernelTraits::SmemLayoutA{}
    );

    Tensor sB = make_tensor(
        make_smem_ptr(reinterpret_cast<DTypeB*>(smem_ + KernelTraits::kSmemSizeA)),
        typename KernelTraits::SmemLayoutB{}
    );

    Tensor sS = make_tensor(
        make_smem_ptr(reinterpret_cast<ScaleType*>(smem_ + KernelTraits::kSmemSizeA + KernelTraits::kSmemSizeB)),
        typename KernelTraits::SmemLayoutS{}
    );

    Tensor sC = make_tensor(
        make_smem_ptr(reinterpret_cast<DTypeC*>(smem_)),
        typename KernelTraits::SmemLayoutCReduce{}
    );

    // gemm config
    typename KernelTraits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tCrA = thr_mma.partition_fragment_A(gA(_, _, 0));
    Tensor tCrB = thr_mma.partition_fragment_B(gB(_, _, 0));
    Tensor acc_C = thr_mma.partition_fragment_C(gC);
    clear(acc_C);

    // copy g2s
    typename KernelTraits::TiledCopyG2SA tiled_copy_A_g2s;
    auto thr_copy_A_g2s = tiled_copy_A_g2s.get_thread_slice(tidx);

    Tensor tAgA = thr_copy_A_g2s.partition_S(gA);
    Tensor tAsA = thr_copy_A_g2s.partition_D(sA);
    Tensor tAgA_pred = thr_copy_A_g2s.partition_S(gA_pred);

    typename KernelTraits::TiledCopyG2SB tiled_copy_B_g2s;
    auto thr_copy_B_g2s = tiled_copy_B_g2s.get_thread_slice(tidx);

    Tensor tBgB = thr_copy_B_g2s.partition_S(gB);
    Tensor tBsB = thr_copy_B_g2s.partition_D(sB);

    typename KernelTraits::TiledCopyG2SS tiled_copy_S_g2s;
    auto thr_copy_S_g2s = tiled_copy_S_g2s.get_thread_slice(tidx);

    Tensor tSgS = thr_copy_S_g2s.partition_S(gS);
    Tensor tSsS = thr_copy_S_g2s.partition_D(sS);

    // copy s2r
    auto tiled_copy_A_s2r = make_tiled_copy_A(typename KernelTraits::CopyAtomS2RA{}, tiled_mma);
    auto thr_copy_A_s2r = tiled_copy_A_s2r.get_thread_slice(tidx);
    auto tiled_copy_B_s2r = make_tiled_copy_B(typename KernelTraits::CopyAtomS2RB{}, tiled_mma);
    auto thr_copy_B_s2r = tiled_copy_B_s2r.get_thread_slice(tidx);

    Tensor tCsA = thr_copy_A_s2r.partition_S(sA);
    Tensor tCsB = thr_copy_B_s2r.partition_S(sB);
    
    Tensor tCrA_view = thr_copy_A_s2r.retile_D(tCrA);
    Tensor tCrB_view = thr_copy_B_s2r.retile_D(tCrB);


#if 0
if (thread0()) {
    print("tAgA layout\n");
    print(tAgA.layout());
    print("\n");

    print("tAsA layout\n");
    print(tAsA.layout());
    print("\n");

    print("tBgB layout\n");
    print(tBgB.layout());
    print("\n");

    print("tBsB layout\n");
    print(tBsB.layout());
    print("\n");

    print("tAgA_pred layout\n");
    print(tAgA.layout());
    print("\n");

    // print("tBgB_pred layout\n");
    // print(tBgB_pred.layout());
    // print("\n");
    
    print("gS layout\n");
    print(gS.layout());
    print("\n");

    print("sS layout\n");
    print(sS.layout());
    print("\n");

    print("tSgS layout\n");
    print(tSgS.layout());
    print("\n");

    print("tSsS layout\n");
    print(tSsS.layout());
    print("\n");
}
#endif


    constexpr int prologue_stages = STAGES == 1 ? 1 : STAGES - 1;
    const int main_loop_k = size<2>(gA); // K维度上需要多少个CTA_K
    const int inner_loop_k = size<2>(tCrA); // 1个warp在K维度上需要进行多少次MMA

    int g2s_ld_i = 0; // 读取gmem的哪个分块
    int g2s_st_i = 0; // 存储在smem的哪个stage
    int s2r_ld_i = 0;
    int mma_k_i = 0; // 一个warp在k维的mma iter

    // prefetch g2s
    cute::copy(
        tiled_copy_S_g2s,
        tSgS,
        tSsS
    );
    
    #pragma unroll
    for (g2s_ld_i = 0; g2s_ld_i < prologue_stages; g2s_ld_i++, g2s_st_i++) {
        copy_g2s<KernelTraits::IS_EVEN_M>(
            tiled_copy_A_g2s,
            tAgA(_, _, _, g2s_ld_i),
            tAsA(_, _, _, g2s_st_i),
            tAgA_pred(_, _, _, g2s_ld_i),
            M
        );
        cute::copy(
            tiled_copy_B_g2s,
            tBgB(_, _, _, g2s_ld_i),
            tBsB(_, _, _, g2s_st_i)
        );
        if constexpr (STAGES > 1)
            cp_async_fence();
    }

    if constexpr (STAGES > 1)
        cp_async_wait<STAGES - 2>();
    __syncthreads();

#if 0
if (thread0()) {
    print("tSgS data\n");
    print_tensor(tSgS);
    print("tSsS data\n");
    print_tensor(tSsS);
    print("\n");
    print("sS data\n");
    print_tensor(sS);
    print("\n");
}
#endif

    // prefetch s2r
    cute::copy(
        tiled_copy_A_s2r,
        tCsA(_, _, mma_k_i, s2r_ld_i),
        tCrA_view(_, _, mma_k_i)
    );
    // cute::copy(
    //     tiled_copy_B_s2r,
    //     tCsB(_, _, mma_k_i, s2r_ld_i),
    //     tCrB_view(_, _, mma_k_i)
    // );

    // 先copy到int8
    Tensor tCrB_i8 = make_tensor<DTypeB>(tCrB(_, _, mma_k_i).layout());
    cute::copy(
        tCsB(_, _, mma_k_i, s2r_ld_i),
        tCrB_i8
    );
    convert_i8x4_to_fp16x4(tCrB_i8, tCrB(_, _, mma_k_i));

#if 0
    __syncthreads();
if (thread0()) {
    printf("main_loop_k %d\n", main_loop_k);
    // print("tCsA data\n");
    // print_tensor(tCsA(_, 0, 0, 0));
    // print("\n");
    print("tCsB data\n");
    print_tensor(tCsB(_, 0, 0, 0));
    print("\n");

    // print("tCrA layout\n");
    // print(tCrA.layout());
    // print("\n");
    print("tCrB layout\n");
    print(tCrB.layout());
    print("\n");

    // print("tCrA data\n");
    // print_tensor(tCrA(_, 0, 0));
    // print("\n");
    print("tCrB data\n");
    print_tensor(tCrB(_, 0, 0));
    print("\n");

    // print("tCrA_view layout\n");
    // print(tCrA_view.layout());
    // print("\n");
    // print("tCrB_view layout\n");
    // print(tCrB_view.layout());
    // print("\n");

    // print("tCrA_view data\n");
    // print_tensor(tCrA_view(_, 0, 0));
    // print("\n");
    // print("tCrB_view data\n");
    // print_tensor(tCrB_view(_, 0, 0));
    // print("\n");

    // print("tCrB Size\n");
    // print(size(tCrB) * sizeof(tCrB(0, 0, 0)));
    // print("\n");

    // constexpr int numel = decltype(size<0>(tCrB))::value;
    // constexpr int dq_loop_reg_n = decltype(size<1, 0>(tCrB))::value;
    // constexpr int dq_loop_mma_n = decltype(size<1, 1>(tCrB))::value;
    // printf("numel %d\n", numel);
    // printf("dq_loop_reg_n %d\n", dq_loop_reg_n);
    // printf("dq_loop_mma_n %d\n", dq_loop_mma_n);
}
#endif

#if 0
__syncthreads();
if (thread0()) {
// if (tidx == 9 && n_block_idx ==0 && m_block_idx == 0) {
    print("tCrB layout\n");
    print(tCrB.layout());
    print("\n");

    print("tCrB data\n");
    print_tensor(tCrB(_, _, 0));
    print("\n");

    print("tCrB_i8 layout\n");
    print(tCrB_i8.layout());
    print("\n");

    print("tCrB_i8 data\n");
    print_tensor(tCrB_i8);
    print("\n");

    // 转换应该没问题，但看能不能支持任意长度（首先看看是否需要）
    // EdqFastNumericArrayConverter<Element, DTypeB, 4> convert_i8s2half;
    // Tensor tCrB_fp16 = make_tensor<Element>(tCrB_i8.layout());
    // for (int i = 0; i < size<1>(tCrB_fp16); i++) {
    //     auto src_ptr = reinterpret_cast<const cutlass::Array<DTypeB, 4>*>(
    //         tCrB_i8(_, i).data());
    //     auto dst_ptr = reinterpret_cast<cutlass::Array<Element, 4>*>(
    //         tCrB_fp16(_, i).data());
    //     *dst_ptr = convert_i8s2half(*src_ptr);
    // }
    // convert_i8x4_to_fp16x4(tCrB_i8, tCrB_fp16);

    // print("tCrB_fp16 layout\n");
    // print(tCrB_fp16.layout());
    // print("\n");

    // print("tCrB_fp16 data\n");
    // print_tensor(tCrB_fp16);
    // print("\n");

    // auto f = convert_i8s2half(*reinterpret_cast<const cutlass::Array<DTypeB, 4> *>(tCrB_i8(_, 0).data()));
    // auto tmp = make_tensor(make_rmem_ptr<cutlass::half_t>(&f), tCrB_i8(_, 0).layout());
    // print("type convert\n");
    // print_tensor(tmp);
    // print("\n");
}
#endif

    dequantize_sym<KernelTraits::NUM_WARP_M, KernelTraits::NUM_WARP_N>(tCrB(_, _, mma_k_i), sS, lane_idx, warp_mn_idx);

#if 0
__syncthreads();
if (thread0()) {
// if (tidx == 9 && n_block_idx ==0 && m_block_idx == 0) {
    print("tCrB layout\n");
    print(tCrB.layout());
    print("\n");

    print("tCrB data\n");
    print_tensor(tCrB(_, _, 0));
    print("\n");
}

#endif

    #pragma unroll 1
    for (int main_loop_i = 0; main_loop_i < main_loop_k; main_loop_i++) {
        #pragma unroll
        for (int inner_loop_i = 0; inner_loop_i < inner_loop_k; inner_loop_i++) {
            int inner_loop_next = (inner_loop_i + 1) % inner_loop_k;

            // pipeline
            if (inner_loop_i == inner_loop_k - 1) {
                if constexpr (STAGES > 1)
                    cp_async_wait<STAGES - 2>();
                __syncthreads();
                if constexpr ((STAGES & (STAGES - 1)) == 0)
                    s2r_ld_i = (s2r_ld_i + 1) & (STAGES - 1);    
                else
                    s2r_ld_i = (s2r_ld_i + 1) % STAGES;
            }

            // s2r next mma
            cute::copy(
                tiled_copy_A_s2r,
                tCsA(_, _, inner_loop_next, s2r_ld_i),
                tCrA_view(_, _, inner_loop_next)
            );
            // cute::copy(
            //     tiled_copy_B_s2r,
            //     tCsB(_, _, inner_loop_next, s2r_ld_i),
            //     tCrB_view(_, _, inner_loop_next)
            // );
            Tensor tCrB_i8 = make_tensor<DTypeB>(tCrB(_, _, inner_loop_next).layout());
            cute::copy(
                tCsB(_, _, inner_loop_next, s2r_ld_i),
                tCrB_i8
            );
            convert_i8x4_to_fp16x4(tCrB_i8, tCrB(_, _, inner_loop_next));
            dequantize_sym<KernelTraits::NUM_WARP_M, KernelTraits::NUM_WARP_N>(tCrB(_, _, inner_loop_next), sS, lane_idx, warp_mn_idx);

            if (inner_loop_i == 0) {
                if (g2s_ld_i < main_loop_k) {
                    copy_g2s<KernelTraits::IS_EVEN_M>(
                        tiled_copy_A_g2s,
                        tAgA(_, _, _, g2s_ld_i),
                        tAsA(_, _, _, g2s_st_i),
                        tAgA_pred(_, _, _, g2s_ld_i),
                        M
                    );
                    cute::copy(
                        tiled_copy_B_g2s,
                        tBgB(_, _, _, g2s_ld_i),
                        tBsB(_, _, _, g2s_st_i)
                    );
                    g2s_ld_i++;
                    if constexpr ((STAGES & (STAGES - 1)) == 0)
                        g2s_st_i = (g2s_st_i + 1) & (STAGES - 1);    
                    else
                        g2s_st_i = (g2s_st_i + 1) % STAGES;
                }
                if constexpr (STAGES > 1)
                    cp_async_fence();
            }

            cute::gemm(
                tiled_mma,
                acc_C,
                tCrA(_, _, inner_loop_i),
                tCrB(_, _, inner_loop_i),
                acc_C
            );

        }
    }

    // type convert from fp32 to fp16
    constexpr int numel_c = decltype(size(acc_C))::value;
    cutlass::NumericArrayConverter<DTypeC, ElementAccum, numel_c> convert_float2half;
    auto frag = convert_float2half(*reinterpret_cast<const cutlass::Array<ElementAccum, numel_c> *>(acc_C.data()));
    Tensor rC = make_tensor(make_rmem_ptr<DTypeC>(&frag), acc_C.layout());

#if 0
__syncthreads();
if (tidx == 255 && m_block_idx == 8 && n_block_idx == 73) {
    print("rC layout\n");
    print(rC.layout());
    print("\n");

    print("rC data\n");
    print_tensor(rC);
    print("\n");
}
#endif

    // r2s C
    auto tiled_copy_C_r2s = make_tiled_copy_C(typename KernelTraits::CopyAtomR2SC{}, tiled_mma);
    auto thr_copy_C_r2s = tiled_copy_C_r2s.get_thread_slice(tidx);

    Tensor tCrAccC = thr_copy_C_r2s.retile_S(rC);
    Tensor tCsAccC = thr_copy_C_r2s.partition_D(sC(_, _, warp_k_idx));

    cute::copy(tiled_copy_C_r2s, tCrAccC, tCsAccC);

    // s2r2g C
    typename KernelTraits::TiledCopyS2GC tiled_copy_C_s2g;
    auto thr_copy_C_s2g = tiled_copy_C_s2g.get_thread_slice(tidx);

    Tensor tCsC = thr_copy_C_s2g.partition_S(sC(_, _, 0));
    Tensor tCgC = thr_copy_C_s2g.partition_D(gC);
    Tensor tCgC_pred = thr_copy_C_s2g.partition_D(gC_pred);

    const int st_m_loop = size<1>(tCgC);
    const int st_n_loop = size<2>(tCgC);

#if 0
if (tidx == 0 && m_block_idx == 5 && n_block_idx == 0) {
// if (thread0()) {
    // print("tCsC layout\n");
    // print(tCsC.layout());
    // print("\n");

    // print("tCgC layout\n");
    // print(tCgC.layout());
    // print("\n");
    print("tCgC_pred data\n");
    print(tCgC_pred(0, 0, 0));
    print("\n");
    print("tCgC_pred data\n");
    print(tCgC_pred(0, 1, 0));
    print("\n");
}
#endif

    __syncthreads();

    #pragma unroll 
    for (int st_m = 0; st_m < st_m_loop; st_m++) {
        if (!NeedCheck || get<0>(tCgC_pred(0, st_m, 0)) < M) {
            #pragma unroll
            for (int st_n = 0; st_n < st_n_loop; st_n++) {
                auto tCrC = make_tensor_like<DTypeC>(tCgC(_, st_m, st_n));
                cute::copy(
                    tiled_copy_C_s2g,
                    tCsC(_, st_m, st_n),
                    tCrC
                );
                // TODO: sym dequant可以考虑放在这里
                cute::copy(
                    tiled_copy_C_s2g,
                    tCrC,
                    tCgC(_, st_m, st_n)
                );
            }
        }
    }

#if 0
__syncthreads();
if (tidx == 255 && m_block_idx == 8 && n_block_idx == 73) {
    print("tCgC layout\n");
    print(tCgC.layout());
    print("\n");

    print("tCgC data\n");
    print_tensor(tCgC(_, 0, 0));
    print("\n");
}
#endif
}


#define LAUNCH_KERNEL                                                           \
    const bool is_even_m = token % CTA_M == 0;                                 \
    BOOL_SWITCH(is_even_m, IS_EVEN_M, [&] {                                    \
        using KTraits = KernelTraits<                                          \
            CTA_M, CTA_N, CTA_K,                                               \
            WARP_M, WARP_N, WARP_K,                                            \
            STAGES, IS_EVEN_M,                                                 \
            cutlass::half_t, DTypeB, cutlass::half_t                           \
        >;                                                                     \
        static constexpr int NUM_WARPS = KTraits::NUM_WARPS;                   \
        static constexpr int kSmemSize = KTraits::kSmemSize;                   \
                                                                               \
        int num_blocks_m = (token + CTA_M - 1) / CTA_M;                        \
        int num_blocks_n = (out_channel + CTA_N - 1) / CTA_N;                  \
                                                                               \
        const int log_tile = get_log_tile<8>(num_blocks_m);                    \
        const int tile_shift = 1 << log_tile;                                  \
                                                                               \
        dim3 grid_dim(num_blocks_n * tile_shift, (num_blocks_m + tile_shift - 1) / tile_shift); \
        dim3 block_dim(KTraits::WARP_SIZE, NUM_WARPS);                         \
                                                                               \
        auto kernel = w8a16_ws_gemm<KTraits>;                                  \
        if (kSmemSize >= 48 * 1024) {                                          \
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize); \
        }                                                                      \
        kernel<<<grid_dim, block_dim, kSmemSize>>>(                            \
            qact, qweight, out_feat,                                           \
            w_scale,                                                           \
            token, out_channel, in_channel                                     \
        );                                                                     \
    });


// Kernel Dispatch
// 确定分块系数和launch的kernel
template <typename DTypeB>
void w8a16_ws_gemm_dispatch(
    cutlass::half_t* __restrict__ qact,
    DTypeB* __restrict__ qweight,
    cutlass::half_t* __restrict__ out_feat,
    cutlass::half_t* __restrict__ w_scale,
    const int token,
    const int out_channel,
    const int in_channel
) {
    const int block_count = (token / 32) * (out_channel / 128);
    if (block_count < 108 * 2) {
        // 每个CTA的分块大小
        static constexpr int CTA_M = 32;
        static constexpr int CTA_N = 128;
        static constexpr int CTA_K = 64;

        // 每个WARP处理在各个维度上处理多大的数据
        static constexpr int WARP_M = 32;
        static constexpr int WARP_N = 32;
        static constexpr int WARP_K = 64;

        // 流水线阶段数
        static constexpr int STAGES = 4;

        LAUNCH_KERNEL
    }
    else if (block_count < 108 * 6){
        // 每个CTA的分块大小
        static constexpr int CTA_M = 32;
        static constexpr int CTA_N = 128;
        static constexpr int CTA_K = 64;

        // 每个WARP处理在各个维度上处理多大的数据
        static constexpr int WARP_M = 32;
        static constexpr int WARP_N = 32;
        static constexpr int WARP_K = 64;

        // 流水线阶段数
        static constexpr int STAGES = 4;

        LAUNCH_KERNEL
    }
    else if (block_count < 108 * 8){
        // 每个CTA的分块大小
        static constexpr int CTA_M = 64;
        static constexpr int CTA_N = 256;
        static constexpr int CTA_K = 64;

        // 每个WARP处理在各个维度上处理多大的数据
        static constexpr int WARP_M = 64;
        static constexpr int WARP_N = 32;
        static constexpr int WARP_K = 64;

        // 流水线阶段数
        static constexpr int STAGES = 4;

        LAUNCH_KERNEL
    }
    else {
        // 每个CTA的分块大小
        static constexpr int CTA_M = 128;
        static constexpr int CTA_N = 256;
        static constexpr int CTA_K = 64;

        // 每个WARP处理在各个维度上处理多大的数据
        static constexpr int WARP_M = 128;
        static constexpr int WARP_N = 32;
        static constexpr int WARP_K = 64;

        // 流水线阶段数
        static constexpr int STAGES = 3;

        LAUNCH_KERNEL
    }

    // // 每个CTA的分块大小
    // static constexpr int CTA_M = 128;
    // static constexpr int CTA_N = 256;
    // static constexpr int CTA_K = 64;

    // // 每个WARP处理在各个维度上处理多大的数据
    // static constexpr int WARP_M = 128;
    // static constexpr int WARP_N = 32;
    // static constexpr int WARP_K = 64;

    // // 流水线阶段数
    // static constexpr int STAGES = 4;

    // const bool is_even_m = token % CTA_M == 0;
    // BOOL_SWITCH(is_even_m, IS_EVEN_M, [&] {
    //     using KTraits = KernelTraits<
    //         CTA_M, CTA_N, CTA_K,
    //         WARP_M, WARP_N, WARP_K,
    //         STAGES, IS_EVEN_M,
    //         cutlass::half_t, DTypeB, cutlass::half_t
    //     >;
        
    //     w8a16_ws_gemm_launch<KTraits>(
    //         qact, qweight, out_feat, 
    //         w_scale,
    //         token, out_channel, in_channel
    //     );
    // });
}


// pybind接口
at::Tensor w8a16_ws_gemm_cuda(
    const at::Tensor &qact, // (seqlen, in_feat)
    const at::Tensor &qweight, // (out_feat, in_feat)
    const at::Tensor &w_scale, // (out_feat)
    std::optional<at::Tensor> &out_ // (seqlen, out_feat)
) {

    const int token = qact.size(0);
    const int in_channel = qact.size(-1);
    const int out_channel = qweight.size(-2);
    TORCH_CHECK(qact.size(-1) == qweight.size(-1));
    TORCH_CHECK(qweight.size(-2) == w_scale.size(-1));
    // printf("w8a16_ws_gemm_cuda\n");

    at::Tensor out_feat;
    if (out_.has_value()) {
        out_feat = out_.value();
        TORCH_CHECK(out_feat.dtype() == at::kHalf);
        TORCH_CHECK(out_feat.size(-2) == qact.size(-2));
        TORCH_CHECK(out_feat.size(-1) == qweight.size(-2));
        TORCH_CHECK(out_feat.is_cuda(), "Out_feat must be on CUDA");
    }
    else {
        out_feat = torch::empty({token, out_channel}, qact.options().dtype(at::kHalf));
    }

    auto qact_ptr = reinterpret_cast<cutlass::half_t*>(qact.data_ptr());
    auto w_scale_ptr = reinterpret_cast<cutlass::half_t*>(w_scale.data_ptr());
    auto out_feat_ptr = reinterpret_cast<cutlass::half_t*>(out_feat.data_ptr());
    DISPATCH_WEIGHT_DTYPE(qweight.scalar_type(), weight_type, {
        auto qweight_ptr = reinterpret_cast<weight_type*>(qweight.data_ptr());
        w8a16_ws_gemm_dispatch(
            qact_ptr, qweight_ptr, out_feat_ptr, 
            w_scale_ptr,
            token, out_channel, in_channel
        );
    });

    return out_feat;
}