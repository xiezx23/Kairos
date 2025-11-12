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

#include "w8a8_wsas_gemm.h"
#include "utils.h"

using namespace cute;

// KernelTraits
template <
    int CTA_M_, int CTA_N_, int CTA_K_,
    int WARP_M_, int WARP_N_, int WARP_K_,
    int STAGES_, bool IS_EVEN_M_,
    typename DTypeA_, typename DTypeB_, typename DTypeC_ // int8_t, int8_t, cutlass::half_t
>
struct KernelTraits {
    // 对于Type，目前考虑int8 GEMM后DQ，因此都是int8，且反量化因子是fp16
 
    // ABD在输入输出时的类型
    using DTypeA = DTypeA_;
    using DTypeB = DTypeB_;
    using DTypeC = DTypeC_;
 
    using Element = int8_t; // GEMM时的Type
    using ElementAccum = int32_t; // 累加的Type
    
    using ScaleType = cutlass::half_t;

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
    using MMA_OP = SM80_16x8x32_S32S8S8S32_TN;
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
    // S=3 -> 2^3组M一行
    // awq的w8a8应该是一次load了两行(2x64)，所以以两行为一组进行swizzle?
    static constexpr int kShmLoadSwizzleB = 3; // 2^3=8 row
    static constexpr int kShmLoadSwizzleM = 4; // 2^4=16 col = 16xint8 = 16Byte = uint128
    // static constexpr int kShmLoadSwizzleS = cutlass::log2_down<CTA_K / (1 << kShmLoadSwizzleM)>::value;
    static constexpr int kShmLoadSwizzleS = 3;

    using SmemLayoutAtomSwizzled = decltype(
        composition(
            Swizzle<kShmLoadSwizzleB, kShmLoadSwizzleM, kShmLoadSwizzleS>{},
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
            SmemLayoutAtomSwizzled{},
            ShapeA{}
        )
    );

    using SmemLayoutB = decltype(
        tile_to_shape(
            SmemLayoutAtomSwizzled{},
            ShapeB{}
        )
    );

    using SmemLayoutAtomC = decltype(
        composition(
            Swizzle<3, 3, 3>{},
            Layout<
                Shape<Int<kMmaPermuteM>, Int<kMmaPermuteN>>,
                Stride<Int<kMmaPermuteN>, _1>
            >{}
        )
    );

    using SmemLayoutC = decltype(tile_to_shape(
        SmemLayoutAtomC{},
        ShapeC{})
    );

    using SmemLayoutCReduce = decltype(tile_to_shape(
        SmemLayoutAtomC{},
        Shape<Int<CTA_M>, Int<CTA_N>, Int<NUM_WARP_K>>{})
    );
    
    static constexpr int kSmemSizeA = size(SmemLayoutA{}) * sizeof(Element);
    static constexpr int kSmemSizeB = size(SmemLayoutB{}) * sizeof(Element);
    static constexpr int kSmemSizeCReduce = size(SmemLayoutCReduce{}) * sizeof(ElementAccum);
    static constexpr int kSmemSize = 
        (kSmemSizeA + kSmemSizeB) > kSmemSizeCReduce 
        ? (kSmemSizeA + kSmemSizeB) 
        : kSmemSizeCReduce;

    // Gmem Copy
    // 每个线程一次load多少个元素(128b / sizeof(int8_t) == 16)
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(CTA_K % kGmemElemsPerLoad == 0, "CTA_K must be a mutiple of kGmemElemsPerLoad");
    // 多少个线程用于load一行
    static constexpr int kGmemThreadsPerRow = CTA_K / kGmemElemsPerLoad; 
    static_assert(NUM_THREADS % kGmemThreadsPerRow == 0, "NUM_THREADS must be a multiple of kGmemThreadsPerRow");
    // static_assert(NUM_THREADS != 0, "error threads");
    // 每列有多少线程进行load
    static constexpr int kGmemThreadsPerCol = NUM_THREADS / kGmemThreadsPerRow;

    using G2SLayoutAtom = Layout<
        Shape<Int<kGmemThreadsPerCol>, Int<kGmemThreadsPerRow>>,
        Stride<Int<kGmemThreadsPerRow>, _1>
    >;

    using CopyAtomG2S = std::conditional_t<
        STAGES == 1,
        Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>,
        Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, Element>
    >;

    using TiledCopyG2S = decltype(
        make_tiled_copy(
            CopyAtomG2S{},
            G2SLayoutAtom{},
            Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}
        )
    );

    // S2R
    using CopyAtomS2R = std::conditional_t<
        std::is_same_v<Element, int8_t>,
        Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<32>, Element>,
        Copy_Atom<SM75_U32x4_LDSM_N, Element>
    >;


    // 写回S时，一次copy2个int32
    using CopyAtomR2S = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<32>, ElementAccum>;

    // 将Smem中的C写到Reg
    using CopyAtomS2RC = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccum>;
    static constexpr int kRegElemsPerLoad = sizeof(cute::uint128_t) / sizeof(ElementAccum); // 4
    // 多少个线程用于store一行
    static constexpr int kRegThreadsPerRow = CTA_N / kRegElemsPerLoad; 
    // 每列有多少线程进行store
    static constexpr int kRegThreadsPerCol = NUM_THREADS / kRegThreadsPerRow;

    using S2RCLayoutAtom = Layout<
        Shape<Int<kRegThreadsPerCol>, Int<kRegThreadsPerRow>>,
        Stride<Int<kRegThreadsPerRow>, _1>
    >;

    using TiledCopyS2RC = decltype(
        make_tiled_copy(
            CopyAtomS2RC{},
            S2RCLayoutAtom{},
            Layout<Shape<_1, Int<kRegElemsPerLoad>>>{}
        )
    );

    // 将Reg中的C写回Gmem
    using CopyAtomR2G = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<64>, DTypeC>;
    static constexpr int kGmemElemsPerStore = sizeof(cute::uint64_t) / sizeof(DTypeC);
    // 多少个线程用于store一行
    static constexpr int kGmemThreadsPerRowC = CTA_N / kGmemElemsPerStore; 
    // 每列有多少线程进行store
    static constexpr int kGmemThreadsPerColC = NUM_THREADS / kGmemThreadsPerRowC;

    using R2GLayoutAtom = Layout<
        Shape<Int<kGmemThreadsPerColC>, Int<kGmemThreadsPerRowC>>,
        Stride<Int<kGmemThreadsPerRowC>, _1>
    >;

    using TiledCopyR2G = decltype(
        make_tiled_copy(
            CopyAtomR2G{},
            R2GLayoutAtom{},
            Layout<Shape<_1, Int<kGmemElemsPerStore>>>{}
        )
    );

};


template <typename KernelTraits>
__global__ void w8a8_wsas_gemm(
    int8_t* __restrict__ A,
    int8_t* __restrict__ B,
    cutlass::half_t* __restrict__ C,
    cutlass::half_t* __restrict__ A_scale,
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
    static constexpr int NUM_WARPS = KernelTraits::NUM_WARPS;
    static constexpr int NUM_THREADS = KernelTraits::NUM_THREADS;
    static constexpr bool SPLIT_K = KernelTraits::SPLIT_K;
    static constexpr int STAGES = KernelTraits::STAGES;

    // TODO: CTA-level split-K
    // TODO: Warp-level split-K（awq的有但没有用）
    const int num_blocks_m = (M + CTA_M - 1) / CTA_M;
    const int num_blocks_n = (N + CTA_N - 1) / CTA_N;

    const int log_tile = get_log_tile<8>(num_blocks_m);
    const uint2 block_idx_mapping = get_block_idx_mapping(blockIdx.x, blockIdx.y, log_tile);
    const int n_block_idx = block_idx_mapping.x;
    const int m_block_idx = block_idx_mapping.y;
    
    // 防止多余的block参与计算
    if (m_block_idx >= num_blocks_m || n_block_idx >= num_blocks_n)
        return;

    const bool NeedCheck = m_block_idx == num_blocks_m - 1 && !KernelTraits::IS_EVEN_M;

    const int lane_idx = threadIdx.x;
    const int warp_idx = threadIdx.y;
    const int warp_k_idx = warp_idx / KernelTraits::NUM_WARPS_MN;
    const int tidx = threadIdx.x + threadIdx.y * blockDim.x;
    // const int tidx = threadIdx.x;

    // Smem分配
    extern __shared__ char smem_[];
    
    // gmem tensor
    // 是gmem tensor那不能用1，应该用Int<1>{}或_1{}
    Tensor mA = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element*>(A)),
        make_shape(M, K),
        make_stride(K, Int<1>{})
    );
    
    Tensor mB = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element*>(B)),
        make_shape(N, K),
        make_stride(K, Int<1>{})
    );
    
    Tensor mScaleA = make_tensor(
        make_gmem_ptr(reinterpret_cast<ScaleType*>(A_scale)),
        make_shape(M),
        make_stride(Int<1>{})
    );

    Tensor mScaleB = make_tensor(
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

    Tensor gScaleA = local_tile(
        mScaleA,
        Shape<Int<CTA_M>>{},
        make_coord(m_block_idx)
    );

    Tensor gScaleB = local_tile(
        mScaleB,
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
        make_smem_ptr(reinterpret_cast<Element*>(smem_)),
        typename KernelTraits::SmemLayoutA{}
    );

    Tensor sB = make_tensor(
        sA.data() + size(sA),
        typename KernelTraits::SmemLayoutB{}
    );

    Tensor sC = make_tensor(
        make_smem_ptr(reinterpret_cast<ElementAccum*>(smem_)),
        typename KernelTraits::SmemLayoutCReduce{}
    );

    // gemm config
    typename KernelTraits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    // reg tensor
    // 这里只会用个Shape，所以是gmem tensor还是smem tensor无所谓
    Tensor tCrA = thr_mma.partition_fragment_A(gA(_, _, 0));
    Tensor tCrB = thr_mma.partition_fragment_B(gB(_, _, 0));
    Tensor acc_C = thr_mma.partition_fragment_C(gC);
    clear(acc_C);

    // copy g2s
    typename KernelTraits::TiledCopyG2S tiled_copy_g2s;
    auto thr_copy_g2s = tiled_copy_g2s.get_thread_slice(tidx);

    Tensor tAgA = thr_copy_g2s.partition_S(gA);
    Tensor tAsA = thr_copy_g2s.partition_D(sA);
    Tensor tAgA_pred = thr_copy_g2s.partition_S(gA_pred);
    
    Tensor tBgB = thr_copy_g2s.partition_S(gB);
    Tensor tBsB = thr_copy_g2s.partition_D(sB);

    // copy s2r
    auto tiled_copy_A_s2r = make_tiled_copy_A(typename KernelTraits::CopyAtomS2R{}, tiled_mma);
    auto thr_copy_A_s2r = tiled_copy_A_s2r.get_thread_slice(tidx);
    auto tiled_copy_B_s2r = make_tiled_copy_B(typename KernelTraits::CopyAtomS2R{}, tiled_mma);
    auto thr_copy_B_s2r = tiled_copy_B_s2r.get_thread_slice(tidx);

    Tensor tCsA = thr_copy_A_s2r.partition_S(sA);
    Tensor tCsB = thr_copy_B_s2r.partition_S(sB);
    
    Tensor tCrA_view = thr_copy_A_s2r.retile_D(tCrA);
    Tensor tCrB_view = thr_copy_B_s2r.retile_D(tCrB);

    constexpr int prologue_stages = STAGES == 1 ? 1 : STAGES - 1;

    const int main_loop_k = size<2>(gA); // K维度上需要多少个CTA_K
    const int inner_loop_k = size<2>(tCrA); // 1个warp在K维度上需要进行多少次MMA

    int g2s_ld_i = 0; // 读取gmem的哪个分块
    int g2s_st_i = 0; // 存储在smem的哪个stage
    int s2r_ld_i = 0;
    int mma_k_i = 0; // 一个warp在k维的mma iter

#if 0
    if (thread0()) {
        // print("\nsmem A size\n");
        // print(cosize(sA.layout()));

        // print("\nsmem B size\n");
        // print(cosize(sB.layout()));

        // print("\nmma\n");
        // print(tiled_mma);

        // print("\ncopy g2s\n");
        // print(tiled_copy_g2s);

        // print("\ngA\n");
        // print(gA.layout());
        // print("\ngB\n");
        // print(gB.layout());
        // print("\ngC\n");
        // print(gC.layout());

        // print("\ntAgA\n");
        // print(tAgA.layout());
        // print("\ntAsA\n");
        // print(tAsA.layout());

        // print("\ntBgB\n");
        // print(tBgB.layout());
        // print("\ntBsB\n");
        // print(tBsB.layout());

        print("\ntCsA\n");
        print(tCsA.layout());
        print("\ntCsB\n");
        print(tCsB.layout());

        print("\ntCrA\n");
        print(tCrA.layout());
        print("\ntCrB\n");
        print(tCrB.layout());

        print("\nacc_C\n");
        print(acc_C.layout());
        print("\n");

        // print("\nfff\n");
        // print(tAgA(_, _, _, 0).layout());
        // print("\n");
        // print(recast<uint128_t>(tAgA(_, _, _, 0)).layout());
        
        // print("\nfff2\n");
        // print(tAsA(_, _, _, 0).layout());
        // print("\n");
        // print(recast<uint128_t>(tAsA(_, _, _, 0)).layout());

        // print("\nint8\n");
        // Tensor i8tensor = make_tensor_like<int8_t>(tAgA(_, _, _, 0));
        // print(i8tensor.layout());
        // print("\n");
        // print(recast<uint128_t>(i8tensor).layout());

        // print("\nuint8\n");
        // Tensor u8tensor = make_tensor_like<uint8_t>(tAgA(_, _, _, 0));
        // print(recast<uint128_t>(u8tensor).layout());

        // print("\nhalf\n");
        // auto f16tensor = make_tensor_like<half_t>(tAgA(_, _, _, 0));
        // print(recast<uint128_t>(f16tensor).layout());
    }

#endif

    // prefetch g2s
    #pragma unroll
    for (g2s_ld_i = 0; g2s_ld_i < prologue_stages; g2s_ld_i++, g2s_st_i++) {
        copy_g2s<KernelTraits::IS_EVEN_M>(
            tiled_copy_g2s,
            tAgA(_, _, _, g2s_ld_i),
            tAsA(_, _, _, g2s_st_i),
            tAgA_pred(_, _, _, g2s_ld_i),
            M
        );
        
        cute::copy(
            tiled_copy_g2s,
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
    if (tidx == 8 && n_block_idx ==0 && m_block_idx == 0) {
        printf("=== S2R Copy Debug ===\n");
        printf("mma_k_i: %d, s2r_ld_i: %d\n", mma_k_i, s2r_ld_i);
        
        // print("tCsA layout:\n");
        // print(tCsA.layout());
        // print("tCrA_view layout:\n");
        // print(tCrA_view.layout());
        
        // print("Source slice layout:\n");
        // print(tCsA(_, _, mma_k_i, s2r_ld_i).layout());
        // print("Dest slice layout:\n");
        // print(tCrA_view(_, _, mma_k_i).layout());
        
        // printf("Source slice size: %d\n", (int)size(tCsA(_, _, mma_k_i, s2r_ld_i)));
        // printf("Dest slice size: %d\n", (int)size(tCrA_view(_, _, mma_k_i)));

        print("gA layout:\n");
        print(gA.layout());
        print("\n");

        print("Data values in tAgA:\n");
        print_tensor(tAgA(_, 0, 0, 0));
        print("\n");

        // print("Data values in tBgB:\n");
        // print_tensor(tBgB);
        // print("\n");

        // print("Data values in tCsA:\n");
        // print_tensor(tCsA);
        // print("\n");

        // print("Data values in tCsB:\n");
        // print_tensor(tCsB);
        // print("\n");
    }

#endif

    // prefetch s2r
    cute::copy(
        tiled_copy_A_s2r,
        tCsA(_, _, mma_k_i, s2r_ld_i),
        tCrA_view(_, _, mma_k_i)
    );
    cute::copy(
        tiled_copy_B_s2r,
        tCsB(_, _, mma_k_i, s2r_ld_i),
        tCrB_view(_, _, mma_k_i)
    );

#if 0
    __syncthreads();
    if(tidx == 4) {

        // print("Data values in tAgA:\n");
        // print_tensor(tAgA);
        // print("\n");
        // print("Data values in tAsA:\n");
        // print_tensor(tAsA);
        // print("\n");
        // print("Data values in tCsA:\n");
        // print_tensor(tCsA);
        // print("\n");
        print("Data values in tCrA_view:\n");
        print_tensor(tCrA_view);
        print("\n");
        // print("Data values in tCrA:\n");
        // print_tensor(tCrA);
        // print("\n");


        // print("Data values in tBgB:\n");
        // print_tensor(tBgB);
        // print("\n");
        // print("Data values in tBsB:\n");
        // print_tensor(tBsB);
        // print("\n");
        // print("Data values in tCsB:\n");
        // print_tensor(tCsB);
        // print("\n");
        print("Data values in tCrB_view:\n");
        print_tensor(tCrB_view);
        print("\n");
    } 
#endif

    #pragma unroll 1
    for (int main_loop_i = 0; main_loop_i < main_loop_k; main_loop_i++) {
        #pragma unroll
        for (int inner_loop_i = 0; inner_loop_i < inner_loop_k; inner_loop_i++) {
            int inner_loop_next = (inner_loop_i + 1) % inner_loop_k;

            // 旧的pipeline
            if (inner_loop_i == inner_loop_k - 1) {
                if constexpr (STAGES > 1)
                    cp_async_wait<STAGES - 2>();
                __syncthreads();
                // 判断是否为2的幂
                if constexpr ((STAGES & (STAGES - 1)) == 0)
                    s2r_ld_i = (s2r_ld_i + 1) & (STAGES - 1);    
                else
                    s2r_ld_i = (s2r_ld_i + 1) % STAGES;
            }

            // s2r
            cute::copy(
                tiled_copy_A_s2r,
                tCsA(_, _, inner_loop_next, s2r_ld_i),
                tCrA_view(_, _, inner_loop_next)
            );
            cute::copy(
                tiled_copy_B_s2r,
                tCsB(_, _, inner_loop_next, s2r_ld_i),
                tCrB_view(_, _, inner_loop_next)
            );

            if (inner_loop_i == 0) {
                if (g2s_ld_i < main_loop_k) {
                    copy_g2s<KernelTraits::IS_EVEN_M>(
                        tiled_copy_g2s,
                        tAgA(_, _, _, g2s_ld_i),
                        tAsA(_, _, _, g2s_st_i),
                        tAgA_pred(_, _, _, g2s_ld_i),
                        M
                    );
                    cute::copy(
                        tiled_copy_g2s,
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

    
#if 0
    __syncthreads();
    if (tidx == 4) {

        print("Data values in acc_C:\n");
        print_tensor(acc_C);
        print("\n");

    } 
#endif

    // Acc_C r2s
    auto tiled_copy_C_r2s = make_tiled_copy_C(typename KernelTraits::CopyAtomR2S{}, tiled_mma);
    auto thr_copy_C_r2s = tiled_copy_C_r2s.get_thread_slice(tidx);

    Tensor tAccCrAccC = thr_copy_C_r2s.retile_S(acc_C);
    Tensor tAccCsAccC = thr_copy_C_r2s.partition_D(sC(_, _, 0));
    
    const int m_loop_r2s = size<1>(tAccCrAccC);
    const int n_loop_r2s = size<2>(tAccCrAccC);

    // r2s copy
    #pragma unroll
    for (int m = 0; m < m_loop_r2s; m++) {
        #pragma unroll
        for (int n = 0; n < n_loop_r2s; n++) {
            cute::copy(
                tiled_copy_C_r2s,
                tAccCrAccC(_, m, n),
                tAccCsAccC(_, m, n)
            );
        }
    }

#if 0
    __syncthreads();
    if (tidx == 4) {

        print("Data values in acc_C:\n");
        print_tensor(acc_C);
        print("\n");

        print("Data values in sC:\n");
        print_tensor(sC);
        print("\n");

    } 
#endif

    // TODO: 进行K维度多个Warp的Reduce
    // if constexpr (KernelTraits::NUM_WARP_K > 1) {
    //     __syncthreads();
    //     // 使用s2g的layout进行reduce
    //     const int kNCPerThread = (CTA_M * CTA_N + NUM_THREADS - 1) / NUM_THREADS; // 8
    //     const int kNThreadsPerRow = CTA_N / kNCPerThread; // 8
    //     ElementAccum* sCReduce = reinterpret_cast<ElementAccum*>(smem_);

    //     for (int i = 0; i < kNCPerThread; i++) {
    //         const int row = tidx / kNThreadsPerRow;
    //         const int col = tidx % kNThreadsPerRow * kNCPerThread + i;
    //         const int offset = row * CTA_N + col;

    //         for (int w = 1; w < KernelTraits::NUM_WARP_K; w++) {
    //             sCReduce[offset] += sCReduce[offset + w * CTA_M * CTA_N];
    //         }
    //     }
        
    // }


    // C s2r, r2g

    typename KernelTraits::TiledCopyS2RC tiled_copy_C_s2r;
    auto thr_copy_C_s2r = tiled_copy_C_s2r.get_thread_slice(tidx);
    Tensor tCsC = thr_copy_C_s2r.partition_S(sC(_, _, 0));

    typename KernelTraits::TiledCopyR2G tiled_copy_C_r2g;
    auto thr_copy_C_r2g = tiled_copy_C_r2g.get_thread_slice(tidx);
    Tensor tCgC = thr_copy_C_r2g.partition_D(gC);
    Tensor tCgC_pred = thr_copy_C_r2g.partition_D(gC_pred);

    __syncthreads();

    static_assert(size<0>(tCsC) == size<0>(tCgC));
    static_assert(size<1>(tCsC) == size<1>(tCgC));
    static_assert(size<2>(tCsC) == size<2>(tCgC));

    constexpr int numel = decltype(size<0>(tCsC))::value;
    cutlass::NumericArrayConverter<float, ElementAccum, numel> convert_int2float;
    cutlass::NumericArrayConverter<DTypeC, float, numel> convert_float2half;
    
#if 0
    if (thread0()) {

        print("\ntCsC layout:\n");
        print(tCsC.layout());
        print("\ntCgC layout:\n");
        print(tCgC.layout());
        print("\ntCgC_pred layout:\n");
        print(tCgC_pred.layout());
        // print("\n");
        // print(get<0>(tCgC_pred(_, 0, 0)));
        print("\n");
    } 
#endif

    // 每行有多少线程进行store
    static constexpr int kRegThreadsPerRow = KernelTraits::kRegThreadsPerRow; 
    // 每列有多少线程进行store
    static constexpr int kRegThreadsPerCol = KernelTraits::kRegThreadsPerCol;


#if 0
    if (thread0()) {

        print("\ntCsC layout:\n");
        print(tCsC.layout());
        print("\ntCgC layout:\n");
        print(tCgC.layout());
        print("\ngScaleA layout:\n");
        print(gScaleA.layout());
        print("\ngScaleB layout:\n");
        print(gScaleB.layout());
        print("\nrC_int layout:\n");
        print(rC_int.layout());
        print("\nrC_fp32 layout:\n");
        print(rC_fp32.layout());
        print("\nrC_half layout:\n");
        print(rC_half.layout());

        print_tensor(gScaleA);
        print_tensor(gScaleB);
        
        print_tensor(rC_int);
        print_tensor(rC_fp32);
        print_tensor(rC_half);

        printf("rC_int data: %d\n", rC_int(0));
        printf("rC_fp32 data: %f\n", rC_fp32(0));



        print("\n");
    } 
#endif
    const int st_m_loop = size<1>(tCsC);
    const int st_n_loop = size<2>(tCgC);
    #pragma unroll
    for (int st_m = 0; st_m < st_m_loop; st_m++) {
        if (!NeedCheck || get<0>(tCgC_pred(0, st_m, 0)) < M) {
            #pragma unroll
            for (int st_n = 0; st_n < st_n_loop; st_n++) {
                // s2r
                Tensor rC_int = make_tensor_like<ElementAccum>(tCsC(_, st_m, st_n));
                cute::copy(
                    tCsC(_, st_m, st_n),
                    rC_int
                );
                
                // Convert from int32 to float
                auto frag = convert_int2float(*reinterpret_cast<const cutlass::Array<ElementAccum, numel> *>(rC_int.data()));
                Tensor rC_fp32 = make_tensor(make_rmem_ptr<float>(&frag), rC_int.layout());

                // dequantize
                int a_scale_idx = st_m * kRegThreadsPerCol + (tidx / kRegThreadsPerRow);
                int b_scale_idx = st_n * kRegThreadsPerRow + (tidx % kRegThreadsPerRow) * numel;
                float a_scale = __half2float(gScaleA(a_scale_idx).to_half());
                half2* b_scale_base_ptr = reinterpret_cast<half2*>(&gScaleB(b_scale_idx));
                float2* c_res_base_ptr = reinterpret_cast<float2*>(&rC_fp32(0));
                constexpr int vec_loop = numel / 2;

                #pragma unroll
                for (int i = 0; i < vec_loop; i++) {
                    half2* b_scale_vec_ptr = b_scale_base_ptr + i;
                    float2* c_res_vec_ptr = c_res_base_ptr + i;
                    float2 b_scale = __half22float2(*b_scale_vec_ptr);
                    c_res_vec_ptr->x *= a_scale * b_scale.x;
                    c_res_vec_ptr->y *= a_scale * b_scale.y;
                }
                
                // Convert from float to half
                auto frag2 = convert_float2half(*reinterpret_cast<const cutlass::Array<float, numel> *>(rC_fp32.data()));
                Tensor rC_half = make_tensor(make_rmem_ptr<half_t>(&frag2), rC_fp32.layout());
                
                // r2g
                cute::copy(
                    rC_half,
                    tCgC(_, st_m, st_n)
                );
            }
        }
    }


#if 0
    if(thread0()) {

        print("\nsC\n");
        print(sC.layout());
        print("\ntAccCrAccC\n");
        print(tAccCrAccC.layout());
        print("\ntAccCsAccC\n");
        print(tAccCsAccC.layout());
        print("\n");
        print("\ntCsC\n");
        print(tCsC.layout());
        print("\ntCgC\n");
        print(tCgC.layout());
        print("\n");
        // print("Data values in sC:\n");
        // for (int i = 0; i < size(tAccCrAccC); ++i) {
        //     printf("tCsC[%d] = %d\n", i, tAccCrAccC(i));
        // }
        // print_tensor(sC);
    } 
#endif

}


#define LAUNCH_KERNEL                                                          \
    const bool is_even_m = token % CTA_M == 0;                                 \
    BOOL_SWITCH(is_even_m, IS_EVEN_M, [&] {                                    \
        using KTraits = KernelTraits<                                          \
            CTA_M, CTA_N, CTA_K,                                               \
            WARP_M, WARP_N, WARP_K,                                            \
            STAGES, IS_EVEN_M,                                                 \
            int8_t, int8_t, half_t                                             \
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
        auto kernel = w8a8_wsas_gemm<KTraits>;                                 \
        if (kSmemSize >= 48 * 1024) {                                          \
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize); \
        }                                                                      \
        kernel<<<grid_dim, block_dim, kSmemSize>>>(                            \
            qact, qweight, out_feat,                                           \
            act_scale, w_scale,                                                \
            token, out_channel, in_channel                                     \
        );                                                                     \
    });


// Kernel Dispatch
// 确定分块系数和launch的kernel
void w8a8_wsas_gemm_dispatch(
    int8_t* __restrict__ qact,
    int8_t* __restrict__ qweight,
    cutlass::half_t* __restrict__ out_feat,
    cutlass::half_t* __restrict__ act_scale,
    cutlass::half_t* __restrict__ w_scale,
    const int token,
    const int out_channel,
    const int in_channel
){
    // TODO:根据token和out_channel指定参数
    // 由于没有实现Split-K 暂不考虑in_channel
    const int block_count = (token / 32) * (out_channel / 128);
    if (block_count < 108 * 2) {
        // 每个CTA的分块大小
        static constexpr int CTA_M = 32;
        static constexpr int CTA_N = 64;
        static constexpr int CTA_K = 64;

        // 每个WARP处理在各个维度上处理多大的数据
        static constexpr int WARP_M = 32;
        static constexpr int WARP_N = 16;
        static constexpr int WARP_K = 64;

        // 流水线阶段数
        static constexpr int STAGES = 8;

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

}

// pybind接口
at::Tensor w8a8_wsas_gemm_cuda(
    const at::Tensor &qact, // (seqlen, in_feat)
    const at::Tensor &qweight, // (out_feat, in_feat)
    const at::Tensor &act_scale, // (seqlen)
    const at::Tensor &w_scale, // (out_feat)
    std::optional<at::Tensor> &out_ // (seqlen, out_feat)
) {

    const int token = qact.size(0);
    const int in_channel = qact.size(-1);
    const int out_channel = qweight.size(-2);
    TORCH_CHECK(qact.size(-1) == qweight.size(-1));
    TORCH_CHECK(qact.size(-2) == act_scale.size(-1));
    TORCH_CHECK(qweight.size(-2) == w_scale.size(-1));
    // printf("w8a8_wsas_gemm_cuda\n");

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

    auto qact_ptr = reinterpret_cast<int8_t*>(qact.data_ptr<int8_t>());
    auto qweight_ptr = reinterpret_cast<int8_t*>(qweight.data_ptr<int8_t>());
    auto act_scale_ptr = reinterpret_cast<cutlass::half_t*>(act_scale.data_ptr());
    auto w_scale_ptr = reinterpret_cast<cutlass::half_t*>(w_scale.data_ptr());
    auto out_feat_ptr = reinterpret_cast<cutlass::half_t*>(out_feat.data_ptr());

    w8a8_wsas_gemm_dispatch(
        qact_ptr, qweight_ptr, out_feat_ptr, 
        act_scale_ptr, w_scale_ptr,
        token, out_channel, in_channel
    );

    return out_feat;
}