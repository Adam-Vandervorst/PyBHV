#ifndef BHV_PACKED_H
#define BHV_PACKED_H

#include <random>
#include <cstring>
#include "shared.h"


namespace bhv {
    std::mt19937_64 rng;

    word_t * empty() {
        return (word_t *) malloc(BYTES);
    }

    word_t * zeros() {
        return (word_t *) calloc(WORDS, sizeof(word_t));
    }

    void rand_into(word_t * x) {
        for (word_iter_t i = 0; i < WORDS; ++i) {
            x[i] = rng();
        }
    }

    word_t * rand() {
        word_t * x = empty();
        for (word_iter_t i = 0; i < WORDS; ++i) {
            x[i] = rng();
        }
        return x;
    }

    bit_iter_t active(word_t * x) {
        bit_iter_t total = 0;
        for (word_iter_t i = 0; i < WORDS; ++i) {
            total += std::popcount(x[i]);
        }
        return total;
    }

    bit_iter_t hamming(word_t * x, word_t * y) {
        bit_iter_t total = 0;
        for (word_iter_t i = 0; i < WORDS; ++i) {
            total += std::popcount(x[i] ^ y[i]);
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
        word_t * x = zeros();

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
}
#endif //BHV_PACKED_H
