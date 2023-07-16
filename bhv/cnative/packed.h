#ifndef BHV_PACKED_H
#define BHV_PACKED_H

#include <random>
#include <cstring>
#include <cassert>
#include <algorithm>
#include "shared.h"
#include <immintrin.h>
#include <ieee754.h>
#include "TurboSHAKEopt/TurboSHAKE.h"
#include "simdpcg.h"


namespace bhv {
    constexpr word_t ONE_WORD = std::numeric_limits<word_t>::max();
    constexpr bit_word_iter_t HALF_BITS_PER_WORD = BITS_PER_WORD/2;
    constexpr word_t HALF_WORD = ONE_WORD << HALF_BITS_PER_WORD;
    constexpr word_t OTHER_HALF_WORD = ~HALF_WORD;

    template <word_t W>
    word_t * const_bhv() {
        static word_t x [WORDS];
        for (word_t & i : x) i = W;
        return x;
    }

    word_t * ZERO = const_bhv<0>();
    word_t * ONE = const_bhv<ONE_WORD>();
    word_t * HALF = const_bhv<HALF_WORD>();

    std::mt19937_64 rng;

    word_t * empty() {
        return (word_t *) malloc(BYTES);
    }

    word_t * zero() {
        return (word_t *) calloc(WORDS, sizeof(word_t));
    }

    word_t * one() {
        word_t * x = empty();
        for (word_iter_t i = 0; i < WORDS; ++i) {
            x[i] = ONE_WORD;
        }
        return x;
    }

    word_t * half() {
        word_t * x = empty();
        for (word_iter_t i = 0; i < WORDS; ++i) {
            x[i] = HALF_WORD;
        }
        return x;
    }

    void swap_halves_into(word_t * x, word_t * target) {
        for (word_iter_t i = 0; i < WORDS; ++i) {
            target[i] = ((x[i] & HALF_WORD) >> HALF_BITS_PER_WORD) | ((x[i] & OTHER_HALF_WORD) << HALF_BITS_PER_WORD);
        }
    }

// TODO separate RNG out in a separate header too

    void rand_into(word_t * x) {
        for (word_iter_t i = 0; i < WORDS; ++i) {
            x[i] = rng();
        }
    }

    avx2_pcg32_random_t key = {
            .state = {_mm256_set_epi64x(0xb5f380a45f908741, 0x88b545898d45385d, 0xd81c7fe764f8966c, 0x44a9a3b6b119e7bc), _mm256_set_epi64x(0x3cb6e04dc22f629, 0x727947debc931183, 0xfbfa8fdcff91891f, 0xb9384fd8f34c0f49)},
            .inc = {_mm256_set_epi64x(0xbf2de0670ac3d03e, 0x98c40c0dc94e71e, 0xf3565f35a8c61d00, 0xd3c83e29b30df640), _mm256_set_epi64x(0x14b7f6e4c89630fa, 0x37cc7b0347694551, 0x4a052322d95d485b, 0x10f3ade77a26e15e)},
            .pcg32_mult_l =  _mm256_set1_epi64x(UINT64_C(0x5851f42d4c957f2d) & 0xffffffffu),
            .pcg32_mult_h = _mm256_set1_epi64x(UINT64_C(0x5851f42d4c957f2d) >> 32)};

    void rand_into_avx2(word_t * x) {
        for (word_iter_t i = 0; i < WORDS; i += 4) {
            _mm256_storeu_si256((__m256i*)(x + i), avx2_pcg32_random_r(&key));
        }
    }

// TODO adopt the same AVX feature control as majority.h and threshold.h

//    avx512_pcg32_random_t key = {
//        .state = _mm512_set_epi64(0xb5f380a45f908741, 0x88b545898d45385d, 0xd81c7fe764f8966c, 0x44a9a3b6b119e7bc, 0x3cb6e04dc22f629, 0x727947debc931183, 0xfbfa8fdcff91891f, 0xb9384fd8f34c0f49),
//        .inc = _mm512_set_epi64(0xbf2de0670ac3d03e, 0x98c40c0dc94e71e, 0xf3565f35a8c61d00, 0xd3c83e29b30df640, 0x14b7f6e4c89630fa, 0x37cc7b0347694551, 0x4a052322d95d485b, 0x10f3ade77a26e15e),
//          .multiplier =  _mm512_set1_epi64(0x5851f42d4c957f2d)};

//    void rand_into_avx512(word_t * x) {
//        for (word_iter_t i = 0; i < WORDS; i += 4) {
//            _mm256_storeu_si256((__m256i*)(x + i), avx512_pcg32_random_r(&key));
//        }
//    }

    void random_into(word_t * x, float_t p) {
        std::bernoulli_distribution gen(p);

        for (word_iter_t i = 0; i < WORDS; ++i) {
            word_t word = 0;
            for (bit_word_iter_t bit_id = 0; bit_id < BITS_PER_WORD; ++bit_id) {
                if (gen(rng))
                    word |= 1UL << bit_id;
            }
            x[i] = word;
        }
    }

// TODO include in benchmark.cpp, along with its brother that does |=
    void rand2_into(word_t * target, int8_t pow) {
        for (word_iter_t i = 0; i < WORDS; ++i) {
            word_t w = rng();
            for (int8_t p = 1; p < pow; ++p) {
                w &= rng();
            }
            target[i] = w;
        }
    }

// Note This could have an AVX-512 implementation with 512-bit float-level log and floor, and probably and equivalent to generate_canonical
// HOWEVER, probably not worth it at the very moment
    template <bool additive>
    void sparse_random_switch_into(word_t * x, float_t prob, word_t * target) {
        double inv_log_not_prob = 1. / std::log(1 - prob);
        size_t skip_count = std::floor(std::log(std::generate_canonical<float_t, 32>(rng)) * inv_log_not_prob);

        for (word_iter_t i = 0; i < WORDS; ++i) {
            word_t word = x[i];
            while (skip_count < BITS_PER_WORD) {
                if constexpr (additive)
                    word |= 1UL << skip_count;
                else
                    word &= ~(1UL << skip_count);
                skip_count += std::floor(std::log(std::generate_canonical<float_t, 32>(rng)) * inv_log_not_prob);
            }
            skip_count -= BITS_PER_WORD;
            target[i] = word;
        }
    }

    void random_into_1tree_sparse(word_t * x, float_t p) {
        if (p < .36)
            return sparse_random_switch_into<true>(ZERO, p, x);
        else if (p > .64)
            return sparse_random_switch_into<false>(ONE, 1.f - p, x);
        else {
            rand_into(x);
            if (p <= .5)
                sparse_random_switch_into<false>(x, 2*(.5f - p), x);
            else
                sparse_random_switch_into<true>(x, 2*(p - .5f), x);
        }
    }

    uint64_t instruction(float frac, uint8_t* to) {
        ieee754_float p = {frac};
        int32_t exponent = IEEE754_FLOAT_BIAS - int(p.ieee.exponent);
        uint64_t instruction = (1 << (23 + exponent)) | p.ieee.mantissa | (1 << 23);
        instruction = instruction >> (_tzcnt_u64(instruction) + 1);
        *to = 63 - _lzcnt_u64(instruction);
        return instruction;
    }

    void random_into_tree_avx2(word_t * x, float_t p) {
        uint8_t to;
        uint64_t instr = instruction(p, &to);

        for (word_iter_t word_id = 0; word_id < WORDS; word_id += 4) {
            __m256i chunk = avx2_pcg32_random_r(&key);

            for (uint8_t i = 0; i < to; ++i)
                if ((instr & (1 << i)) >> i)
                    chunk = _mm256_or_si256(chunk, avx2_pcg32_random_r(&key));
                else
                    chunk = _mm256_and_si256(chunk, avx2_pcg32_random_r(&key));

            _mm256_storeu_si256((__m256i*)(x + word_id), chunk);
        }
    }

    word_t * rand() {
        word_t * x = empty();
        rand_into(x);
        return x;
    }

    word_t * random(float_t p) {
        word_t * x = empty();
        random_into(x, p);
        return x;
    }

    bit_iter_t active(word_t * x) {
        bit_iter_t total = 0;
        for (word_iter_t i = 0; i < WORDS; ++i) {
            total += __builtin_popcountl(x[i]);
        }
        return total;
    }

    bit_iter_t hamming(word_t * x, word_t * y) {
        bit_iter_t total = 0;
        for (word_iter_t i = 0; i < WORDS; ++i) {
            total += __builtin_popcountl(x[i] ^ y[i]);
        }
        return total;
    }

    bool eq(word_t * x, word_t * y) {
        for (word_iter_t i = 0; i < WORDS; ++i) {
            if (x[i] != y[i])
                return false;
        }
        return true;
    }


    void xor_into(word_t * x, word_t * y, word_t * target) {
        for (word_iter_t i = 0; i < WORDS; ++i) {
            target[i] = x[i] ^ y[i];
        }
    }

    void and_into(word_t * x, word_t * y, word_t * target) {
        for (word_iter_t i = 0; i < WORDS; ++i) {
            target[i] = x[i] & y[i];
        }
    }

    void or_into(word_t * x, word_t * y, word_t * target) {
        for (word_iter_t i = 0; i < WORDS; ++i) {
            target[i] = x[i] | y[i];
        }
    }

    void invert_into(word_t * x, word_t * target) {
        for (word_iter_t i = 0; i < WORDS; ++i) {
            target[i] = ~x[i];
        }
    }

    void select_into(word_t * cond, word_t * when1, word_t * when0, word_t * target) {
        for (word_iter_t i = 0; i < WORDS; ++i) {
            target[i] = when0[i] ^ (cond[i] & (when0[i] ^ when1[i]));
        }
    }

    #include "threshold.h"

    #include "majority.h"

    #include "representative.h"

    #include "permutation.h"

    void rehash_into(word_t * x, word_t * target) {
        TurboSHAKE(512, (uint8_t *)x, BYTES, 0x1F, (uint8_t *)target, BYTES);
    }
}
#endif //BHV_PACKED_H
