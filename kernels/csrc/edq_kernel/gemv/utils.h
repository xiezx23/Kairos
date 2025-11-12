#pragma once
#include <cuda_fp16.h>

template <typename T, typename S, int N>
struct EdqFastNumericArrayConverter {};

// No Interleaved, No Biased
// assume that input type is uint8_t and origin type is uint8_t
template <>
struct EdqFastNumericArrayConverter<half, uint8_t, 4> {
    using target_type = float2; // half * 4
    using source_type = uint32_t; // int8_t * 4

    __inline__ __device__ static target_type convert(source_type const& source) {
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

    __inline__ __device__ target_type operator()(source_type const& source) {
        return convert(source);
    }
};


// No Interleaved, assume that input type is int8_t
template <>
struct EdqFastNumericArrayConverter<half, int8_t, 4> {
    using target_type = float2; // half * 4
    using source_type = uint32_t; // int8_t * 4

    __inline__ __device__ static target_type convert(source_type const& source) {
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

    __inline__ __device__ target_type operator()(source_type const& source) {
        return convert(source);
    }
};


// Interleaved, assume that input type is int4(4b)
// convert 8x4b(int32) to 8xhalf(uint4(means 4 int))
template <>
struct EdqFastNumericArrayConverter<uint4, int, 8> {
    using target_type = uint4; // half * 8
    using source_type = int; // int4(4b) * 8

    __inline__ __device__ static target_type convert(source_type const& source) {
        target_type result;

        int*      h   = reinterpret_cast<int*>(&result);
        int const i4s = reinterpret_cast<int const&>(source);

        int const i4s_high = i4s >> 8;

        constexpr int LOW_MASK = 0x000f000f;
        constexpr int HIGH_MASK = 0x00f000f0;
        constexpr int MAGIC_NUM = 0x64006400;
        constexpr int LOP3_LUT = (0xf0 & 0xcc) | 0xaa;

        // 获取第0和第4个, Interleave后为0 1
        // (i4s & 0x000f000f) | 0x64006400
        __asm__ __volatile__(
            "lop3.b32 %0, %1, %2, %3, %4;\n"
            : "=r"(h[0])
            : "r"(i4s), "n"(LOW_MASK), "n"(MAGIC_NUM), "n"(LOP3_LUT)
        );

        // 获取第1和第5个, Interleave后为2 3
        // (i4s & 0x00f000f0) | 0x64006400
        __asm__ __volatile__(
            "lop3.b32 %0, %1, %2, %3, %4;\n"
            : "=r"(h[1])
            : "r"(i4s), "n"(HIGH_MASK), "n"(MAGIC_NUM), "n"(LOP3_LUT)
        );

        // 获取第2和第6个, Interleave后为4 5
        // ((i4s >> 8) & 0x000f000f) | 0x64006400
        __asm__ __volatile__(
            "lop3.b32 %0, %1, %2, %3, %4;\n"
            : "=r"(h[2])
            : "r"(i4s_high), "n"(LOW_MASK), "n"(MAGIC_NUM), "n"(LOP3_LUT)
        );

        // 获取第3和第7个, Interleave后为6 7
        // ((i4s >> 8) & 0x00f000f0) | 0x64006400
        __asm__ __volatile__(
            "lop3.b32 %0, %1, %2, %3, %4;\n"
            : "=r"(h[3])
            : "r"(i4s_high), "n"(HIGH_MASK), "n"(MAGIC_NUM), "n"(LOP3_LUT)
        );

        const int SUB = 0x64006400; // -1024 得到uint4
        const int MUL = 0x2c002c00; // 1/16用于处理高4bit的数据 使其尾数的开始位回到最低位
        const int ADD = 0xd400d400; // -64 = -1024/16 因为这里要用fma

        // 快速反量化为正确数值
        *reinterpret_cast<half2*>(&h[0]) = __hsub2(
            *reinterpret_cast<half2*>(&h[0]),
            *reinterpret_cast<const half2*>(&SUB)
        );
        
        *reinterpret_cast<half2*>(&h[1]) = __hfma2(
            *reinterpret_cast<half2*>(&h[1]),
            *reinterpret_cast<const half2*>(&MUL),
            *reinterpret_cast<const half2*>(&ADD)
        );

        *reinterpret_cast<half2*>(&h[2]) = __hsub2(
            *reinterpret_cast<half2*>(&h[2]),
            *reinterpret_cast<const half2*>(&SUB)
        );

        *reinterpret_cast<half2*>(&h[3]) = __hfma2(
            *reinterpret_cast<half2*>(&h[3]),
            *reinterpret_cast<const half2*>(&MUL),
            *reinterpret_cast<const half2*>(&ADD)
        );

        return result;
    }

    __inline__ __device__ target_type operator()(source_type const& source) {
        return convert(source);
    }
};


#define BATCH_SWITCH(BS, CONST_NAME, ...)      \
  [&] {                                         \
    if (BS == 1) {                              \
      constexpr static int CONST_NAME = 1;      \
      return __VA_ARGS__();                     \
    } else if (BS == 2) {                       \
      constexpr static int CONST_NAME = 2;      \
      return __VA_ARGS__();                     \
    } else if (BS == 3) {                       \
      constexpr static int CONST_NAME = 3;      \
      return __VA_ARGS__();                     \
    } else if (BS == 4) {                       \
      constexpr static int CONST_NAME = 4;      \
      return __VA_ARGS__();                     \
    } else if (BS == 5) {                       \
      constexpr static int CONST_NAME = 5;      \
      return __VA_ARGS__();                     \
    } else if (BS == 6) {                       \
      constexpr static int CONST_NAME = 6;      \
      return __VA_ARGS__();                     \
    } else if (BS == 7) {                       \
      constexpr static int CONST_NAME = 7;      \
      return __VA_ARGS__();                     \
    } else if (BS == 8) {                       \
      constexpr static int CONST_NAME = 8;      \
      return __VA_ARGS__();                     \
    } else if (BS == 9) {                       \
      constexpr static int CONST_NAME = 9;      \
      return __VA_ARGS__();                     \
    } else if (BS == 10) {                       \
      constexpr static int CONST_NAME = 10;      \
      return __VA_ARGS__();                     \
    } else if (BS == 11) {                       \
      constexpr static int CONST_NAME = 11;      \
      return __VA_ARGS__();                     \
    } else if (BS == 12) {                       \
      constexpr static int CONST_NAME = 12;      \
      return __VA_ARGS__();                     \
    } else if (BS == 13) {                       \
      constexpr static int CONST_NAME = 13;      \
      return __VA_ARGS__();                     \
    } else if (BS == 14) {                       \
      constexpr static int CONST_NAME = 14;      \
      return __VA_ARGS__();                     \
    } else if (BS == 15) {                       \
      constexpr static int CONST_NAME = 15;      \
      return __VA_ARGS__();                     \
    } else if (BS == 16) {                       \
      constexpr static int CONST_NAME = 16;      \
      return __VA_ARGS__();                     \
    } else if (BS == 17) {                       \
      constexpr static int CONST_NAME = 17;      \
      return __VA_ARGS__();                     \
    } else if (BS == 18) {                       \
      constexpr static int CONST_NAME = 18;      \
      return __VA_ARGS__();                     \
    } else if (BS == 19) {                       \
      constexpr static int CONST_NAME = 19;      \
      return __VA_ARGS__();                     \
    } else if (BS == 20) {                       \
      constexpr static int CONST_NAME = 20;      \
      return __VA_ARGS__();                     \
    } else if (BS == 21) {                       \
      constexpr static int CONST_NAME = 21;      \
      return __VA_ARGS__();                     \
    } else if (BS == 22) {                       \
      constexpr static int CONST_NAME = 22;      \
      return __VA_ARGS__();                     \
    } else if (BS == 23) {                       \
      constexpr static int CONST_NAME = 23;      \
      return __VA_ARGS__();                     \
    } else if (BS == 24) {                       \
      constexpr static int CONST_NAME = 24;      \
      return __VA_ARGS__();                     \
    } else if (BS == 25) {                       \
      constexpr static int CONST_NAME = 25;      \
      return __VA_ARGS__();                     \
    } else if (BS == 26) {                       \
      constexpr static int CONST_NAME = 26;      \
      return __VA_ARGS__();                     \
    } else if (BS == 27) {                       \
      constexpr static int CONST_NAME = 27;      \
      return __VA_ARGS__();                     \
    } else if (BS == 28) {                       \
      constexpr static int CONST_NAME = 28;      \
      return __VA_ARGS__();                     \
    } else if (BS == 29) {                       \
      constexpr static int CONST_NAME = 29;      \
      return __VA_ARGS__();                     \
    } else if (BS == 30) {                       \
      constexpr static int CONST_NAME = 30;      \
      return __VA_ARGS__();                     \
    } else if (BS == 31) {                       \
      constexpr static int CONST_NAME = 31;      \
      return __VA_ARGS__();                     \
    } else if (BS == 32) {                       \
      constexpr static int CONST_NAME = 32;      \
      return __VA_ARGS__();                     \
    } else {                                    \
      throw std::runtime_error("Unsupport batch size > 32 for GEMV kernel.\n"); \
    }                                           \
  }()