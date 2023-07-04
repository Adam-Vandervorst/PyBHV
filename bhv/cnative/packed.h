#ifndef BHV_PACKED_H
#define BHV_PACKED_H

#include <random>
#include <cstring>
#include <algorithm>
#include "shared.h"
#include <immintrin.h>
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

// USES _mm256_cmpgt_epu8_mask is AVX-512
//    void byte_threshold_into(word_t ** xs, uint8_t size, uint8_t threshold, word_t* dst) {
//        __m256i threshold_simd = _mm256_set1_epi8(threshold);
//        uint8_t** xs_bytes = (uint8_t**)xs;
//        uint8_t* dst_bytes = (uint8_t*)dst;
//
//        for (byte_iter_t byte_id = 0; byte_id < BYTES; byte_id += 4) {
//            __m256i total_simd = _mm256_set1_epi8(0);
//
//            for (uint8_t i = 0; i < size; ++i) {
//                uint8_t* bytes_i = xs_bytes[i];
//                uint64_t spread_words[4] = {
//
//                        //GOAT, use this for 11 < N < 54
////                        deposit_bits_lut[bytes_i[byte_id]],
////                        deposit_bits_lut[bytes_i[byte_id + 1]],
////                        _pdep_u64(bytes_i[byte_id + 2], 0x0101010101010101),
////                        _pdep_u64(bytes_i[byte_id + 3], 0x0101010101010101)
//
//                        // //GOAT, use this for N > 53
//                         _pdep_u64(bytes_i[byte_id], 0x0101010101010101),
//                         _pdep_u64(bytes_i[byte_id + 1], 0x0101010101010101),
//                         _pdep_u64(bytes_i[byte_id + 2], 0x0101010101010101),
//                         _pdep_u64(bytes_i[byte_id + 3], 0x0101010101010101)
//
//                };
//
//                total_simd = _mm256_adds_epu8(total_simd, *((__m256i *)spread_words));
//            }
//
//            *((uint32_t*)&(dst_bytes[byte_id])) = _mm256_cmpgt_epu8_mask(total_simd, threshold_simd);
//        }
//    }

    template <uint8_t size>
    void logic_majority_into(word_t ** xs, word_t* dst) {
        uint8_t half = size/2;
        __m256i grid [size/2 + 1][size/2 + 1];

        for (word_iter_t word_id = 0; word_id < WORDS; word_id += 4) {
            grid[half][half] = _mm256_loadu_si256((__m256i *)(xs[size - 1] + word_id));

            for (uint8_t i = 0; i < half; ++i) {
                __m256i chunk = _mm256_loadu_si256((__m256i *)(xs[size - i - 2] + word_id));
                grid[half - i - 1][half] = grid[half - i][half] & chunk;
                grid[half][half - i - 1] = grid[half][half - i] | chunk;
            }

            for (uint8_t i = half - 1; i < half; --i) for (uint8_t j = half - 1; j < half; --j) {
                __m256i chunk = _mm256_loadu_si256((__m256i *)(xs[i + j] + word_id));
                grid[i][j] = grid[i][j + 1] ^ (chunk & (grid[i][j + 1] ^ grid[i + 1][j]));
            }

            _mm256_storeu_si256((__m256i*)(dst + word_id), grid[0][0]);
        }
    }

    void dynamic_logic_majority_into(word_t ** xs, uint8_t size, word_t* dst) {
        switch (size) {
            case 3: logic_majority_into<3>(xs, dst); break;
            case 5: logic_majority_into<5>(xs, dst); break;
            case 7: logic_majority_into<7>(xs, dst); break;
            case 9: logic_majority_into<9>(xs, dst); break;
            case 11: logic_majority_into<11>(xs, dst); break;
            case 13: logic_majority_into<13>(xs, dst); break;
            case 15: logic_majority_into<15>(xs, dst); break;
            case 17: logic_majority_into<17>(xs, dst); break;
            case 19: logic_majority_into<19>(xs, dst); break;
            case 21: logic_majority_into<21>(xs, dst); break;
            case 23: logic_majority_into<23>(xs, dst); break;
            case 25: logic_majority_into<25>(xs, dst); break;
            case 27: logic_majority_into<27>(xs, dst); break;
            case 29: logic_majority_into<29>(xs, dst); break;
            case 31: logic_majority_into<31>(xs, dst); break;
            case 33: logic_majority_into<33>(xs, dst); break;
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

    void select_into(word_t * cond, word_t * when1, word_t * when0, word_t * target) {
        for (word_iter_t i = 0; i < WORDS; ++i) {
            target[i] = when0[i] ^ (cond[i] & (when0[i] ^ when1[i]));
        }
    }

    #include "threshold.h"

    #include "majority.h"

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
