#ifndef BHV_CORE_H
#define BHV_CORE_H

#include <bit>
#include <functional>
#include <random>
#include <cstring>
#include <cassert>
#include <algorithm>
#include <immintrin.h>
#include "shared.h"
#ifdef __AVX2__
#include "simdpcg.h"
#endif
#ifdef __AVX512__
#include "TurboSHAKE_AVX512/TurboSHAKE.h"
#else
#include "TurboSHAKE_opt/TurboSHAKE.h"
#endif


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

    inline word_t *empty() {
        return (word_t *) aligned_alloc(64, BYTES);
    }

    word_t *zero() {
        word_t * e = empty();
        memset(e, 0, BYTES);
        return e;
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

    void unpack_into(word_t *data, bool *target) {
        for (word_iter_t word_id = 0; word_id < WORDS; ++word_id) {
            bit_iter_t offset = word_id * BITS_PER_WORD;
            word_t word = data[word_id];
            for (bit_word_iter_t bit_id = 0; bit_id < BITS_PER_WORD; ++bit_id) {
                target[offset + bit_id] = word & (1ULL << bit_id);
            }
        }
    }

    void pack_into(bool *data, word_t *target) {
        for (word_iter_t word_id = 0; word_id < WORDS; ++word_id) {
            bit_iter_t offset = word_id * BITS_PER_WORD;
            word_t word = 0;
            for (bit_word_iter_t bit_id = 0; bit_id < BITS_PER_WORD; ++bit_id) {
                if (data[offset + bit_id])
                    word |= 1ULL << bit_id;
            }
            target[word_id] = word;
        }
    }

    inline bool get(word_t *d, bit_iter_t i) {
        return (d[i/BITS_PER_WORD] >> (i % BITS_PER_WORD)) & 1;
    }

    inline void toggle(word_t *d, bit_iter_t i) {
        d[i/BITS_PER_WORD] ^= (1ULL << (i % BITS_PER_WORD));
    }


    #include "io.h"

    #include "ternary.h"

    #include "distance.h"

    #include "random.h"

    #include "threshold.h"

    #include "majority.h"

    #include "representative.h"

    #include "window.h"

    #include "permutation.h"

    #include "hash.h"

    #include "optimization.h"
}
#endif //BHV_CORE_H
