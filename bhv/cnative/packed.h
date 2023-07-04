#ifndef BHV_PACKED_H
#define BHV_PACKED_H

#include <random>
#include <cstring>
#include <cassert>
#include <algorithm>
#include "shared.h"
#include <immintrin.h>
#include "TurboSHAKEopt/TurboSHAKE.h"


namespace bhv {
    constexpr word_t ONE_WORD = std::numeric_limits<word_t>::max();
    constexpr bit_word_iter_t HALF_BITS_PER_WORD = BITS_PER_WORD/2;
    constexpr word_t HALF_WORD = ONE_WORD << HALF_BITS_PER_WORD;
    constexpr word_t OTHER_HALF_WORD = ~HALF_WORD;

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

    void rand_into(word_t * x) {
        for (word_iter_t i = 0; i < WORDS; ++i) {
            x[i] = rng();
        }
    }

    void random_into(word_t * x, float_t p) {
        std::uniform_real_distribution<float> gen(0.0, 1.0);

        for (word_iter_t i = 0; i < WORDS; ++i) {
            word_t word = 0;
            for (bit_word_iter_t bit_id = 0; bit_id < BITS_PER_WORD; ++bit_id) {
                if (gen(rng) < p)
                    word |= 1UL << bit_id;
            }
            x[i] = word;
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
