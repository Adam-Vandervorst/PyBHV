#ifndef BHV_PACKED_H
#define BHV_PACKED_H

#include <random>
#include <cstring>
#include <algorithm>
#include "shared.h"
//#include "TurboSHAKE.h"
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

    template <typename N>
    N* generic_counts(word_t ** xs, N size) {
        N* totals = (N *) calloc(BITS, sizeof(N));

        for (N i = 0; i < size; ++i) {
            word_t * x = xs[i];

            for (word_iter_t word_id = 0; word_id < WORDS; ++word_id) {
                bit_iter_t offset = word_id * BITS_PER_WORD;
                word_t word = x[word_id];
                for (bit_word_iter_t bit_id = 0; bit_id < BITS_PER_WORD; ++bit_id) {
                    totals[offset + bit_id] += ((word >> bit_id) & 1);
                }
            }
        }

        return totals;
    }

    template <typename N>
    word_t* generic_gt(N * totals, N threshold) {
        word_t * x = empty();

        for (word_iter_t word_id = 0; word_id < WORDS; ++word_id) {
            bit_iter_t offset = word_id * BITS_PER_WORD;
            word_t word = 0;
            for (bit_word_iter_t bit_id = 0; bit_id < BITS_PER_WORD; ++bit_id) {
                if (threshold < totals[offset + bit_id])
                    word |= 1UL << bit_id;
            }
            x[word_id] = word;
        }
        free(totals);
        return x;
    }

    word_t * representative_impl(word_t ** xs, size_t size) {
        word_t * x = zero();

        std::uniform_int_distribution<size_t> gen(0, size - 1);
        for (word_iter_t word_id = 0; word_id < WORDS; ++word_id) {
            word_t word = 0;
            for (bit_word_iter_t bit_id = 0; bit_id < BITS_PER_WORD; ++bit_id) {
                size_t x_id = gen(rng);
                if ((xs[x_id][word_id] >> bit_id) & 1)
                    word |=  1UL << bit_id;
            }
            x[word_id] = word;
        }

        return x;
    }

    word_t * n_representatives_impl(word_t ** xs, size_t size) {
        word_t * x = zero();

        std::uniform_int_distribution<size_t> gen(0, size - 1);
        for (word_iter_t word_id = 0; word_id < WORDS; ++word_id) {
            word_t word = 0;
            for (bit_word_iter_t bit_id = 0; bit_id < BITS_PER_WORD; ++bit_id) {
                size_t x_id = gen(rng);
                word |=  1UL << (xs[x_id][word_id] >> bit_id) & 1;
            }
            x[word_id] = word;
        }

        return x;
    }

    word_t* threshold(word_t ** xs, size_t size, size_t threshold) {
        if (size < UINT8_MAX)
            return generic_gt<uint8_t>(generic_counts<uint8_t>(xs, size), threshold);
        else if (size < UINT16_MAX)
            return generic_gt<uint16_t>(generic_counts<uint16_t>(xs, size), threshold);
        else
            return generic_gt<uint32_t>(generic_counts<uint32_t>(xs, size), threshold);
    }

    void select_into(word_t * cond, word_t * when1, word_t * when0, word_t * target) {
        for (word_iter_t i = 0; i < WORDS; ++i) {
            target[i] = when0[i] ^ (cond[i] & (when0[i] ^ when1[i]));
        }
    }

    void majority3_into(word_t * x, word_t * y, word_t * z, word_t * target) {
        for (word_iter_t i = 0; i < WORDS; ++i) {
            target[i] = ((x[i] & y[i]) | (x[i] & z[i]) | (y[i] & z[i]));
        }
    }

    word_t* true_majority(word_t ** xs, size_t size) {
        if (size == 0) return rand();
        else if (size == 1) { word_t * r = empty(); memcpy(r, xs[0], BYTES); return r; }
        else if (size == 2) { word_t * r = rand(); select_into(r, xs[0], xs[1], r); return r; }
        else if (size == 3) { word_t * r = empty(); majority3_into(xs[0], xs[1], xs[2], r); return r; }
        else return threshold(xs, size, size/2);
    }

    word_t* representative(word_t ** xs, size_t size) {
        if (size == 0) return rand();
        else if (size == 1) { word_t * r = empty(); memcpy(r, xs[0], BYTES); return r; }
        else if (size == 2) { word_t * r = rand(); select_into(r, xs[0], xs[1], r); return r; }
        else return representative_impl(xs, size);
    }

    void permute_words_into(word_t * x, word_iter_t* word_permutation, word_t * target) {
        for (word_iter_t i = 0; i < WORDS; ++i) {
            target[i] = x[word_permutation[i]];
        }
    }

    void inverse_permute_words_into(word_t * x, word_iter_t* word_permutation, word_t * target) {
        for (word_iter_t i = 0; i < WORDS; ++i) {
            target[word_permutation[i]] = x[i];
        }
    }

    word_iter_t* rand_word_permutation(uint32_t seed) {
        std::minstd_rand0 perm_rng(seed);

        auto p = (word_iter_t *) malloc(sizeof(word_iter_t)*WORDS);

        for (word_iter_t i = 0; i < WORDS; ++i)
            p[i] = i;

        std::shuffle(p, p + WORDS, perm_rng);

        return p;
    }

    void permute_into(word_t * x, int32_t perm, word_t * target) {
        if (perm == 0) *target = *x;
        else if (perm > 0) permute_words_into(x, rand_word_permutation(perm), target);
        else inverse_permute_words_into(x, rand_word_permutation(-perm), target);
    }

    void rehash_into(word_t * x, word_t * target) {
        TurboSHAKE(512, (uint8_t *)x, BYTES, 0x1F, (uint8_t *)target, BYTES);
    }
}
#endif //BHV_PACKED_H
