#ifndef BHV_CORE_H
#define BHV_CORE_H

#include <bit>
#include <random>
#include <cstring>
#include <cassert>
#include <algorithm>
#include "shared.h"
#include <immintrin.h>
#include "TurboSHAKEopt/TurboSHAKE.h"
#include "simdpcg.h"
#include "platform_compatibility.h"

namespace bhv {
    constexpr word_t ONE_WORD = std::numeric_limits<word_t>::max();
    constexpr bit_word_iter_t HALF_BITS_PER_WORD = BITS_PER_WORD / 2;
    constexpr word_t HALF_WORD = ONE_WORD << HALF_BITS_PER_WORD;
    constexpr word_t OTHER_HALF_WORD = ~HALF_WORD;

    template<word_t W>
    word_t *const_bhv() {
        static word_t x[WORDS];
        for (word_t &i: x) i = W;
        return x;
    }

    word_t *ZERO = const_bhv<0>();
    word_t *ONE = const_bhv<ONE_WORD>();
    word_t *HALF = const_bhv<HALF_WORD>();

    std::mt19937_64 rng;

    word_t *empty() {
        return (word_t *) malloc(BYTES);
    }

    word_t *zero() {
        return (word_t *) calloc(WORDS, sizeof(word_t));
    }

    word_t *one() {
        word_t *x = empty();
        for (word_iter_t i = 0; i < WORDS; ++i) {
            x[i] = ONE_WORD;
        }
        return x;
    }

    word_t *half() {
        word_t *x = empty();
        for (word_iter_t i = 0; i < WORDS; ++i) {
            x[i] = HALF_WORD;
        }
        return x;
    }

    void swap_halves_into(word_t *x, word_t *target) {
        for (word_iter_t i = 0; i < WORDS; ++i) {
            target[i] = ((x[i] & HALF_WORD) >> HALF_BITS_PER_WORD) | ((x[i] & OTHER_HALF_WORD) << HALF_BITS_PER_WORD);
        }
    }

    bool eq(word_t *x, word_t *y) {
        for (word_iter_t i = 0; i < WORDS; ++i) {
            if (x[i] != y[i])
                return false;
        }
        return true;
    }

    void xor_into(word_t *x, word_t *y, word_t *target) {
        for (word_iter_t i = 0; i < WORDS; ++i) {
            target[i] = x[i] ^ y[i];
        }
    }

    void and_into(word_t *x, word_t *y, word_t *target) {
        for (word_iter_t i = 0; i < WORDS; ++i) {
            target[i] = x[i] & y[i];
        }
    }

    void or_into(word_t *x, word_t *y, word_t *target) {
        for (word_iter_t i = 0; i < WORDS; ++i) {
            target[i] = x[i] | y[i];
        }
    }

    void invert_into(word_t *x, word_t *target) {
        for (word_iter_t i = 0; i < WORDS; ++i) {
            target[i] = ~x[i];
        }
    }

    void select_into_reference(word_t *cond, word_t *when1, word_t *when0, word_t *target) {
        for (word_iter_t i = 0; i < WORDS; ++i) {
            target[i] = when0[i] ^ (cond[i] & (when0[i] ^ when1[i]));
        }
    }

#if __AVX512F__
/// @note Under GCC -O3 the references implementation compiles into the same instruction
    void select_into_ternary_avx512(word_t *cond, word_t *when1, word_t *when0, word_t *target) {
        __m512i *cond_vec = (__m512i *)cond;
        __m512i *when1_vec = (__m512i *)when1;
        __m512i *when0_vec = (__m512i *)when0;
        __m512i *target_vec = (__m512i *)target;

        for (word_iter_t i = 0; i < BITS/512; ++i) {
            _mm512_storeu_si512(target_vec + i,
                                _mm512_ternarylogic_epi64(_mm512_loadu_si512(cond_vec + i),
                                                          _mm512_loadu_si512(when1_vec + i),
                                                          _mm512_loadu_si512(when0_vec + i), 0xca));
        }
    }
#endif

#if __AVX512F__
#define select_into select_into_ternary_avx512
#else
#define select_into select_into_reference
#endif

    #include "distance.h"

    #include "random.h"

    #include "threshold.h"

    #include "majority.h"

    #include "representative.h"

    #include "permutation.h"

    void rehash_into(word_t *x, word_t *target) {
        TurboSHAKE(512, (uint8_t *) x, BYTES, 0x1F, (uint8_t *) target, BYTES);
    }
}
#endif //BHV_CORE_H
