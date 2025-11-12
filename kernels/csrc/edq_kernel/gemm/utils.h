#pragma once

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

#include <cuda_fp16.h>

#include <cute/tensor.hpp>

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

#include <torch/extension.h>

using namespace cute;

// 要注意对block进行检查，多余的block不要运行
// 后面要看看有没有更好的CTA Swizzle方式
// 如果采用这个，N应该是一个可以调整的参数
template <int N>
__inline__ __host__ __device__ int get_log_tile(int n) {
  if (N >= 8 && n >= 6)
    return 3;
  else if (N >= 4 && n >= 3)
    return 2;
  else if (N >= 2 && n >= 2)
    return 1;
  else
    return 0;
}


__inline__ __device__ uint2 get_block_idx_mapping(int blockIdx_x,
                                                  int blockIdx_y,
                                                  int log_tile) {
  return make_uint2((blockIdx_x >> log_tile),
                    (blockIdx_y << log_tile) +
                        ((blockIdx_x) & ((1 << (log_tile)) - 1))); // <==> mod 2^log_tile
}


// TODO: 使用向量化进行优化
template <
    int NUM_WARP_M, int NUM_WARP_N,
    typename Tensor0, typename Tensor1
>
__inline__ __device__ void dequantize_sym(
    Tensor0 &&tCrB, // B Reg tensor((2, 2), (frag_n, mma_n), mma_k = 0/1/...)，这里只处理一个k的
    Tensor1 const &sS, // Scale Smem tensor(N, )
    int lane_idx, int warp_idx // threadIdx.x, threadIdx.y(m,n dim)
) {
    // dequantize
    constexpr int numel = decltype(size<0>(tCrB))::value;
    constexpr int dq_loop_reg_n = decltype(size<1, 0>(tCrB))::value;
    constexpr int dq_loop_mma_n = decltype(size<1, 1>(tCrB))::value;

    static_assert(numel % 2 == 0, "numel must be even for half2 optimization");
    
    #pragma unroll
    for (int reg_n_i = 0; reg_n_i < dq_loop_reg_n; reg_n_i++) {
        #pragma unroll
        for (int mma_n_i = 0; mma_n_i < dq_loop_mma_n; mma_n_i++) {
            // 这个索引的计算与mma layout有关
            int scale_col = lane_idx / 4 +
                reg_n_i * 8 +
                warp_idx / NUM_WARP_M * 16 +
                mma_n_i * NUM_WARP_N * 16;
            
            // vectorize
            half scale = sS(scale_col).to_half();
            half2 scale_broadcast = __halves2half2(scale, scale);

            constexpr int vec_loop = numel / 2;
            half2* base_ptr = reinterpret_cast<half2*>(&tCrB(0, make_coord(reg_n_i, mma_n_i)));
            #pragma unroll
            for (int i = 0; i < vec_loop; i++) {
                half2* vec_ptr = base_ptr + i;
                *vec_ptr = __hmul2(*vec_ptr, scale_broadcast);
            }
        }
    }
}


template <
    int NUM_WARP_M, int NUM_WARP_N,
    typename Tensor0, typename Tensor1, typename Tensor2
>
__inline__ __device__ void dequantize_asym(
    Tensor0 &&tCrB, // B Reg tensor((2, 2), (frag_n, mma_n), mma_k = 0/1/...)，这里只处理一个k的
    Tensor1 const &sS, // Scale Smem tensor(N, )
    Tensor2 const &sZ, // Zero point Smem tensor(N, )
    int lane_idx, int warp_idx // threadIdx.x, threadIdx.y(m,n dim)
) {
    // dequantize
    constexpr int numel = decltype(size<0>(tCrB))::value;
    constexpr int dq_loop_reg_n = decltype(size<1, 0>(tCrB))::value;
    constexpr int dq_loop_mma_n = decltype(size<1, 1>(tCrB))::value;

    static_assert(numel % 2 == 0, "numel must be even for half2 optimization");
    
    #pragma unroll
    for (int reg_n_i = 0; reg_n_i < dq_loop_reg_n; reg_n_i++) {
        #pragma unroll
        for (int mma_n_i = 0; mma_n_i < dq_loop_mma_n; mma_n_i++) {
            // 这个索引的计算与mma layout有关
            int scale_col = lane_idx / 4 +
                reg_n_i * 8 +
                warp_idx / NUM_WARP_M * 16 +
                mma_n_i * NUM_WARP_N * 16;

            // vectorize
            half scale = sS(scale_col).to_half();
            half zp = sZ(scale_col).to_half();

            half2 scale_broadcast = __halves2half2(scale, scale);
            half2 zp_broadcast = __hneg2(__halves2half2(zp, zp));
            
            constexpr int vec_loop = numel / 2;
            half2* base_ptr = reinterpret_cast<half2*>(&tCrB(0, make_coord(reg_n_i, mma_n_i)));
            #pragma unroll
            for (int i = 0; i < vec_loop; i++) {
                half2* vec_ptr = base_ptr + i;
                *vec_ptr = __hmul2(__hadd2(*vec_ptr, zp_broadcast), scale_broadcast);
            }
        }
    }
}


template <
    bool is_even_MN=false,
    typename TiledCopy,
    typename Engine0, typename Layout0, 
    typename Engine1, typename Layout1, 
    typename Engine2, typename Layout2
>
__forceinline__ __device__ void copy_g2s(
    TiledCopy tiled_copy,
    Tensor<Engine0, Layout0> const &S,
    Tensor<Engine1, Layout1> &&D,
    Tensor<Engine2, Layout2> const &Pred,
    const int max_MN
) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));

    const int mn_loop = size<1>(S);

    if constexpr (!is_even_MN) {
        #pragma unroll
        for (int mn = 0; mn < mn_loop; ++mn) {
            // MN维度规则或要拷贝的数据在MN维度的坐标没有超出范围
            if (get<0>(Pred(0, mn, 0)) < max_MN) {
                    copy(tiled_copy, S(_, mn, _), D(_, mn, _)); 
            }
        }
    }
    else {
        copy(tiled_copy, S, D); 
    }
};


// Fast Numeric Conversion, refer to FastTransformer
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h
template <typename T, typename S, int N>
struct EdqFastNumericArrayConverter {};


// No Interleaved, No Biased
// assume that input type is uint8_t and origin type is uint8_t
template <>
struct EdqFastNumericArrayConverter<cutlass::half_t, uint8_t, 4> {
    // static_assert(!(N % 4), "N must be multiple of 4.");
    using target_type = cutlass::Array<cutlass::half_t, 4>;
    using source_type = cutlass::Array<uint8_t, 4>;

    CUTLASS_DEVICE
    static target_type convert(source_type const& source) {
        target_type result;

        uint32_t*      h   = reinterpret_cast<uint32_t*>(&result);
        uint32_t const i8u = reinterpret_cast<uint32_t const&>(source);

        // construct fp16x2 from uint8
        static constexpr uint32_t mask_for_elt_01     = 0x5150;
        static constexpr uint32_t mask_for_elt_23     = 0x5352;
        static constexpr uint32_t start_byte_for_fp16 = 0x64646464;
        __asm__ __volatile__(
            "prmt.b32 %0,%1,%2,%3;\n"
            : "=r"(h[0])
            : "r"(i8u), "n"(start_byte_for_fp16), "n"(mask_for_elt_01)
        );
        __asm__ __volatile__(
            "prmt.b32 %0,%1,%2,%3;\n"
            : "=r"(h[1])
            : "r"(i8u), "n"(start_byte_for_fp16), "n"(mask_for_elt_23)
        );

        // assume orgin data type is uint8_t
        static constexpr uint32_t I8s_TO_F16s_MAGIC_NUM = 0x64006400;
        __asm__ __volatile__(
            "sub.f16x2 %0, %1, %2;\n"
            : "=r"(h[0])
            : "r"(h[0]), "r"(I8s_TO_F16s_MAGIC_NUM)
        );
        __asm__ __volatile__(
            "sub.f16x2 %0, %1, %2;\n"
            : "=r"(h[1])
            : "r"(h[1]), "r"(I8s_TO_F16s_MAGIC_NUM)
        );

        return result;
    }

    CUTLASS_DEVICE
    target_type operator()(source_type const& source) {
        return convert(source);
    }
};


// No Interleaved, assume that input type is int8_t
template <>
struct EdqFastNumericArrayConverter<cutlass::half_t, int8_t, 4> {
    // static_assert(!(N % 4), "N must be multiple of 4.");
    using target_type = cutlass::Array<cutlass::half_t, 4>;
    using source_type = cutlass::Array<int8_t, 4>;

    CUTLASS_DEVICE
    static target_type convert(source_type const& source) {
        target_type result;

        uint32_t*      h   = reinterpret_cast<uint32_t*>(&result);
        uint32_t const i8s = reinterpret_cast<uint32_t const&>(source);

        // add 128 convert int8 to uint8
        static constexpr uint32_t add_num = 0x80808080;
        uint32_t const i8u = __vadd4(i8s, add_num);

        // construct fp16x2 from uint8
        static constexpr uint32_t mask_for_elt_01     = 0x5150;
        static constexpr uint32_t mask_for_elt_23     = 0x5352;
        static constexpr uint32_t start_byte_for_fp16 = 0x64646464;
        __asm__ __volatile__(
            "prmt.b32 %0,%1,%2,%3;\n"
            : "=r"(h[0])
            : "r"(i8u), "n"(start_byte_for_fp16), "n"(mask_for_elt_01)
        );
        __asm__ __volatile__(
            "prmt.b32 %0,%1,%2,%3;\n"
            : "=r"(h[1])
            : "r"(i8u), "n"(start_byte_for_fp16), "n"(mask_for_elt_23)
        );

        // orgin data type is int8
        static constexpr uint32_t I8s_TO_F16s_MAGIC_NUM = 0x64806480;
        __asm__ __volatile__(
            "sub.f16x2 %0, %1, %2;\n"
            : "=r"(h[0])
            : "r"(h[0]), "r"(I8s_TO_F16s_MAGIC_NUM)
        );
        __asm__ __volatile__(
            "sub.f16x2 %0, %1, %2;\n"
            : "=r"(h[1])
            : "r"(h[1]), "r"(I8s_TO_F16s_MAGIC_NUM)
        );

        return result;
    }

    CUTLASS_DEVICE
    target_type operator()(source_type const& source) {
        return convert(source);
    }
};

template <
    typename Engine0, typename Layout0, 
    typename Engine1, typename Layout1
>
__forceinline__ __device__ void convert_i8x4_to_fp16x4(
    Tensor<Engine0, Layout0> const& source_tensor,
    Tensor<Engine1, Layout1> && target_tensor
) {

    using SourceType = typename Engine0::value_type;
    using TargetType = typename Engine1::value_type;
    
    EdqFastNumericArrayConverter<TargetType, SourceType, 4> converter;
    
    const int convert_loop = size<1>(source_tensor);
    CUTE_STATIC_ASSERT_V(size<0>(source_tensor) == size<0>(target_tensor));
    CUTE_STATIC_ASSERT_V(size<1>(source_tensor) == size<1>(target_tensor));
    
    #pragma unroll
    for (int i = 0; i < convert_loop; i++) {
        auto src_array = reinterpret_cast<const cutlass::Array<SourceType, 4>*>(
            source_tensor(_, i).data());
        auto dst_array = reinterpret_cast<cutlass::Array<TargetType, 4>*>(
            target_tensor(_, i).data());
        *dst_array = converter(*src_array);
    }
}


#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
  [&] {                                         \
    if (COND) {                                 \
      constexpr static bool CONST_NAME = true;  \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static bool CONST_NAME = false; \
      return __VA_ARGS__();                     \
    }                                           \
  }()

#define DISPATCH_WEIGHT_DTYPE(TorchType, WeightType, ...)          \
  if (TorchType == at::ScalarType::Char) {                         \
    using WeightType = int8_t;                                     \
    __VA_ARGS__                                                    \
  } else if (TorchType == at::ScalarType::Byte) {                  \
    using WeightType = uint8_t;                                    \
    __VA_ARGS__                                                    \
  } else {                                                         \
    TORCH_CHECK(false, "Failed to dispatch weight data type");     \
  }
