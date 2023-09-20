#ifndef BHV_CORE_H
#define BHV_CORE_H

#include <omp.h>
#include <bit>
#include <functional>
#include <random>
#include <cstring>
#include <cassert>
#include <algorithm>
#include "shared.h"
#include <immintrin.h>
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
    std::mt19937_64 rngs [16] = {std::mt19937_64(0),
                                std::mt19937_64(1),
                                std::mt19937_64(2),
                                std::mt19937_64(3),
                                std::mt19937_64(4),
                                std::mt19937_64(5),
                                std::mt19937_64(6),
                                std::mt19937_64(7),
                                std::mt19937_64(8),
                                std::mt19937_64(9),
                                std::mt19937_64(10),
                                std::mt19937_64(11),
                                std::mt19937_64(12),
                                std::mt19937_64(13),
                                std::mt19937_64(14),
                                std::mt19937_64(15)};

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

    #include "ternary.h"

    #include "distance.h"

    #include "random.h"

    #include "threshold.h"

    #include "majority.h"

    #include "representative.h"

    #include "permutation.h"

    #include "hash.h"


    template <typename N>
    N opt_independent(word_t* x, std::function<N (word_t*)> loss) {
        N best_l = loss(x);
        for (bit_iter_t i = 0; i < BITS; ++i) {
            toggle(x, i);
            N l = loss(x);
            if (l > best_l)
                toggle(x, i);
            else
                best_l = l;
        }
        return best_l;
    }

    template <typename N, double U, double L, uint64_t half_time>
    float_t time_based_decay(uint64_t time_step, N loss) {
        double decay_rate = log((U - L)/(U + L))/(double)half_time;
        return L + (U - L)*exp(decay_rate*(double)time_step);
    }

    template <typename N>
    N opt_linear_search(word_t* x, std::function<N (word_t*)> loss,
                        uint64_t max_iter = 10000000,
                        N break_at = std::numeric_limits<N>::min(),
                        std::function<float_t (uint64_t, N)> update = time_based_decay<N, 0.0001, 0.0001, 100000>) {
        N best_l = loss(x);
        N l;
        word_t change [WORDS];
        word_t buf [WORDS];
        bool x_best = true;
        float_t delta = update(0, best_l);

        for (uint64_t it = 1; it <= max_iter; ++it) {
//            std::cout << "delta: " << delta << std::endl;
            random_into(change, delta);

            if (x_best) {
                xor_into(x, change, buf);
                l = loss(buf);
            } else {
                xor_into(buf, change, x);
                l = loss(x);
            }

            if (l < best_l) {
                std::cout << "#";
                best_l = l;
                x_best = not x_best;
            } else {
                std::cout << ".";
            }

            if (l <= break_at) {
                std::cout << std::endl << "it: " << it << std::endl;
                std::cout << "loss: " << best_l << std::endl;
                break;
            }

            delta = update(it, l);
        }

        if (not x_best)
            memcpy(x, buf, BYTES);

        return best_l;
    }

    void thread_init() {
        std::cout << "init " << omp_get_thread_num() << std::endl;
        avx2_key  = {
                .state = {_mm256_set_epi64x(0xb5f380a45f908741, 0x88b545898d45385d, 0xd81c7fe764f8966c, 0x44a9a3b6b119e7bc) * omp_get_thread_num(),
                          _mm256_set_epi64x(0x3cb6e04dc22f629, 0x727947debc931183, 0xfbfa8fdcff91891f, 0xb9384fd8f34c0f49) * omp_get_thread_num()},
                .inc = {_mm256_set_epi64x(0xbf2de0670ac3d03e, 0x98c40c0dc94e71e, 0xf3565f35a8c61d00, 0xd3c83e29b30df640),
                        _mm256_set_epi64x(0x14b7f6e4c89630fa, 0x37cc7b0347694551, 0x4a052322d95d485b, 0x10f3ade77a26e15e)},
                .pcg32_mult_l =  _mm256_set1_epi64x(UINT64_C(0x5851f42d4c957f2d) & 0xffffffffu),
                .pcg32_mult_h = _mm256_set1_epi64x(UINT64_C(0x5851f42d4c957f2d) >> 32)};
    }

    void parallel_init() {
        #pragma omp parallel num_threads(8)
        {
            thread_init();
        }
    }

    template <typename N, uint64_t pool>
    N opt_parallel_search(word_t* x, std::function<N (word_t*)> loss,
                        uint64_t max_iter = 100000,
                        N break_at = std::numeric_limits<N>::min(),
                        std::function<float_t (uint64_t, N)> update = time_based_decay<N, 0.0001, 0.0001, 100000>) {
        N best_l = loss(x);
        N l [pool];
        word_t change [pool][WORDS];
        word_t buf [pool][WORDS];
        bool mask [pool];
        float_t delta = 0.0001;

        for (uint64_t it = 1; it <= max_iter; ++it) {
//            #pragma omp parallel for num_threads(8)
            for (uint64_t p = 0; p < pool; ++p) {
                random_into(change[p], delta);
                xor_into(x, change[p], buf[p]);
                l[p] = loss(buf[p]);
                mask[p] = l[p] < best_l;
            }

//            #pragma omp single
            {
            std::vector<word_t*> matching;
            for (uint64_t p = 0; p < pool; ++p)
                if (mask[p])
                    matching.push_back(buf[p]);

//            std::cout << matching.size() << " matching" << std::endl;

            if (matching.size() % 2 == 0)
                matching.push_back(x);

            true_majority_into(matching.data(), 1, x);

            N tmp = best_l;
            best_l = loss(x);

//            std::cout << (int)tmp - (int)best_l << " loss decrease" << std::endl;

            if (best_l <= break_at) {
                std::cout << std::endl << "it: " << it << std::endl;
                std::cout << "loss: " << best_l << std::endl;
                break;
            }
            };
        }

        return best_l;
    }
}
#endif //BHV_CORE_H
