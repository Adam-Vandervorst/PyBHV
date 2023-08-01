#include <iostream>
#include <chrono>
#include <functional>

#include "core.h"

using namespace std;

#define DO_VALIDATION true

#define MAJ_INPUT_HYPERVECTOR_COUNT 1000001
#define INPUT_HYPERVECTOR_COUNT 1000

#define THRESHOLD
#define MAJ
#define RAND
#define RAND2
#define RANDOM
#define PERMUTE
#define ACTIVE
#define HAMMING
#define INVERT
#define SWAP_HALVES
#define REHASH
#define AND
#define OR
#define XOR
#define SELECT
#define MAJ3
#define TERNARY


float threshold_benchmark(size_t n, size_t threshold, float af, bool display, bool keep_in_cache) {
    //For the simple cases, like 3 vectors, we want a lot of tests to get a reliable number
    //but allocating 2,000 vectors * 10,000 tests starts to exceed resident memory and we end
    //up paying disk swap penalties.  Therefore we do fewer tests in the case with more hypervectors
    const size_t test_count = MAJ_INPUT_HYPERVECTOR_COUNT / n;
    const size_t input_output_count = (keep_in_cache ? 1 : test_count);

    //Init n random vectors for each test
    word_t ***inputs = (word_t***)malloc(sizeof(word_t**) * input_output_count);
    for (size_t i = 0; i < input_output_count; i++) {
        word_t **rs = (word_t **) malloc(sizeof(word_t **) * n);
        for (size_t j = 0; j < n; ++j) {
            rs[j] = bhv::random(af);
        }
        inputs[i] = rs;
    }

    //Allocate a buffer for TEST_COUNT results
    word_t *result_buffer = (word_t *) malloc(input_output_count * BYTES);

    // Gotta assign the result to a volatile, so the test operation doesn't get optimized away
    volatile word_t something = 0;
    volatile word_t something_else = 0;

    auto t1 = chrono::high_resolution_clock::now();
    for (size_t i = 0; i < test_count; i++) {
        const size_t io_buf_idx = (keep_in_cache ? 0 : i);

        word_t *m = result_buffer + (io_buf_idx * BYTES / sizeof(word_t));
        word_t **rs = inputs[io_buf_idx];

        bhv::threshold_into(rs, n, threshold, m);

        // So the test operation doesn't get optimized away
        something = something ^ m[0] + 3 * m[4] + 5 * m[WORDS / 2] + 7 * m[WORDS - 1];
    }
    auto t2 = chrono::high_resolution_clock::now();

    const char* validation_status;
    if (DO_VALIDATION) {
        for (size_t i = 0; i < test_count; i++) {
            const size_t io_buf_idx = (keep_in_cache ? 0 : i);

            word_t *m = result_buffer + (io_buf_idx * BYTES / sizeof(word_t));
            word_t **rs = inputs[io_buf_idx];

            bhv::threshold_into_reference(rs, n, threshold, m);

            something_else = something_else ^ m[0] + 3 * m[4] + 5 * m[WORDS / 2] + 7 * m[WORDS - 1];
        }
        validation_status = ((something == something_else) ? "equiv: √, " : "equiv: X, ");
    } else {
        validation_status = "";
    }

    float mean_test_time = (float) chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count() / (float) test_count;
    if (display)
        cout << n << " hypervectors, " << threshold << " threshold, " << validation_status << "in_cache: " << keep_in_cache
            << ", total: " << mean_test_time / 1000.0 << "µs, normalized: " << mean_test_time / (float) n << "ns/vec" << endl;

    //Clean up our mess
    for (size_t i = 0; i < input_output_count; i++) {
        word_t **rs = inputs[i];
        for (size_t j = 0; j < n; ++j) {
            free(rs[j]);
        }
        free(rs);
    }
    free(result_buffer);
    free(inputs);

    return mean_test_time;
}

float majority_benchmark(size_t n, bool display, bool keep_in_cache) {
    //For the simple cases, like 3 vectors, we want a lot of tests to get a reliable number
    //but allocating 2,000 vectors * 10,000 tests starts to exceed resident memory and we end
    //up paying disk swap penalties.  Therefore we do fewer tests in the case with more hypervectors
    const size_t test_count = MAJ_INPUT_HYPERVECTOR_COUNT / n;
    const size_t input_output_count = (keep_in_cache ? 1 : test_count);

    //Init n random vectors for each test
    word_t ***inputs = (word_t***)malloc(sizeof(word_t**) * input_output_count);
    for (size_t i = 0; i < input_output_count; i++) {
        word_t **rs = (word_t **) malloc(sizeof(word_t **) * n);
        for (size_t j = 0; j < n; ++j) {
            rs[j] = bhv::rand();
        }
        inputs[i] = rs;
    }

    //Allocate a buffer for TEST_COUNT results
    word_t *result_buffer = (word_t *) malloc(input_output_count * BYTES);

    // Gotta assign the result to a volatile, so the test operation doesn't get optimized away
    volatile word_t something = 0;
    volatile word_t something_else = 0;

    auto t1 = chrono::high_resolution_clock::now();
    for (size_t i = 0; i < test_count; i++) {
        const size_t io_buf_idx = (keep_in_cache ? 0 : i);

        word_t *m = result_buffer + (io_buf_idx * BYTES / sizeof(word_t));
        word_t **rs = inputs[io_buf_idx];

        bhv::true_majority_into(rs, n, m);

        // So the test operation doesn't get optimized away
        something = something ^ m[0] + 3 * m[4] + 5 * m[WORDS / 2] + 7 * m[WORDS - 1];
    }
    auto t2 = chrono::high_resolution_clock::now();

    const char* validation_status;
    if (DO_VALIDATION) {
        for (size_t i = 0; i < test_count; i++) {
            const size_t io_buf_idx = (keep_in_cache ? 0 : i);

            word_t *m = result_buffer + (io_buf_idx * BYTES / sizeof(word_t));
            word_t **rs = inputs[io_buf_idx];

            bhv::threshold_into_reference(rs, n, n/2, m);

            something_else = something_else ^ m[0] + 3 * m[4] + 5 * m[WORDS / 2] + 7 * m[WORDS - 1];
        }
        validation_status = ((something == something_else) ? "equiv: √, " : "equiv: X, ");
    } else {
        validation_status = "";
    }

    float mean_test_time = (float) chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count() / (float) test_count;
    if (display)
        cout << n << " hypervectors, " << validation_status << "in_cache: " << keep_in_cache << ", total: " << mean_test_time / 1000.0
             << "µs, normalized: " << mean_test_time / (float) n << "ns/vec" << endl;

    //Clean up our mess
    for (size_t i = 0; i < input_output_count; i++) {
        word_t **rs = inputs[i];
        for (size_t j = 0; j < n; ++j) {
            free(rs[j]);
        }
        free(rs);
    }
    free(result_buffer);
    free(inputs);

    return mean_test_time;
}


float rand_benchmark(bool display, bool keep_in_cache) {
    const int test_count = INPUT_HYPERVECTOR_COUNT;
    const int input_output_count = (keep_in_cache ? 1 : test_count);

    //Allocate a buffer for TEST_COUNT results
    word_t *result_buffer = (word_t *) malloc(input_output_count * BYTES);

    // Gotta assign the result to a volatile, so the test operation doesn't get optimized away
    volatile word_t something = 0;

    auto t1 = chrono::high_resolution_clock::now();
    for (size_t i = 0; i < test_count; ++i) {
        const size_t io_buf_idx = (keep_in_cache ? 0 : i);

        word_t *m = result_buffer + (io_buf_idx * BYTES / sizeof(word_t));

        bhv::rand_into_reference(m);

        something = something ^ m[0] + 3 * m[4] + 5 * m[WORDS / 2] + 7 * m[WORDS - 1];
    }
    auto t2 = chrono::high_resolution_clock::now();

    float mean_test_time = (float) chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count() / (float) test_count;
    if (display)
        cout << "in_cache: " << keep_in_cache << ", total: " << mean_test_time / 1000.0 << "µs" << endl;

    //Clean up our mess
    free(result_buffer);

    return mean_test_time;
}

float rand2_benchmark(bool display, bool keep_in_cache, int pow) {
    const int test_count = INPUT_HYPERVECTOR_COUNT;
    const int input_output_count = (keep_in_cache ? 1 : test_count);
    volatile int p = pow;

    //Allocate a buffer for TEST_COUNT results
    word_t *result_buffer = (word_t *) malloc(input_output_count * BYTES);

    double observed_frac[test_count];

    auto t1 = chrono::high_resolution_clock::now();
    for (size_t i = 0; i < test_count; ++i) {
        const size_t io_buf_idx = (keep_in_cache ? 0 : i);

        word_t *m = result_buffer + (io_buf_idx * BYTES / sizeof(word_t));

        bhv::rand2_into(m, p);

        // once runtime of random_into drops under 500ns, consider removing this
        observed_frac[i] = (double) bhv::active(m) / (double) BITS;
    }
    auto t2 = chrono::high_resolution_clock::now();

    float mean_test_time = (float) chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count() / (float) test_count;
    std::sort(observed_frac, observed_frac + test_count);
    double mean_observed =
            std::reduce(observed_frac, observed_frac + test_count, 0., std::plus<double>()) / (double) test_count;
    mean_observed = mean_observed > .5 ? 1 + std::log2(1 - mean_observed) : -1 - std::log2(mean_observed);
    if (display)
        cout << pow << "pow, observed: " << mean_observed << ", in_cache: " << keep_in_cache << ", total: "
             << mean_test_time / 1000.0 << "µs" << endl;

    //Clean up our mess
    free(result_buffer);

    return mean_test_time;
}


float random_benchmark(bool display, bool keep_in_cache, float base_frac, bool randomize = false) {
    const int test_count = INPUT_HYPERVECTOR_COUNT;
    const int input_output_count = (keep_in_cache ? 1 : test_count);
    volatile double n = randomize ? ((double) (rand() - RAND_MAX / 2) / (double) (RAND_MAX)) / 1000. : 0;
    float p = base_frac + n;

    //Allocate a buffer for TEST_COUNT results
    word_t *result_buffer = (word_t *) malloc(input_output_count * BYTES);

    double observed_frac[test_count];

    auto t1 = chrono::high_resolution_clock::now();
    for (size_t i = 0; i < test_count; ++i) {
        const size_t io_buf_idx = (keep_in_cache ? 0 : i);

        word_t *m = result_buffer + (io_buf_idx * BYTES / sizeof(word_t));

//        bhv::random_into_reference(m, p);
        bhv::random_into_tree_sparse_avx2(m, p);
//        bhv::random_into_buffer_avx2(m, p);

        // once runtime of random_into drops under 500ns, consider removing this
        observed_frac[i] = (double) bhv::active(m) / (double) BITS;
    }
    auto t2 = chrono::high_resolution_clock::now();

    float mean_test_time = (float) chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count() / (float) test_count;
    std::sort(observed_frac, observed_frac + test_count);
    double mean_observed_frac =
            std::reduce(observed_frac, observed_frac + test_count, 0., std::plus<double>()) / (double) test_count - n;
    if (display)
        cout << base_frac << "frac, observed: " << abs(base_frac - mean_observed_frac) << ", in_cache: "
             << keep_in_cache << ", total: " << mean_test_time / 1000.0 << "µs" << endl;

    //Clean up our mess
    free(result_buffer);

    return mean_test_time;
}

template<bool keep_in_cache, int different_permutations>
float permute_benchmark(bool display) {
    const int test_count = INPUT_HYPERVECTOR_COUNT * different_permutations * 2;

    bool correct = true;
    double total_test_time = 0.;
    int perms [different_permutations];
    for (size_t i = 0; i < different_permutations; ++i)
        perms[i] = std::abs(std::rand());

    if constexpr (keep_in_cache) {
        word_t *forward [INPUT_HYPERVECTOR_COUNT][different_permutations + 1];
        word_t *backward [INPUT_HYPERVECTOR_COUNT][different_permutations + 1];
        // different_permutations=3
        // forward:  R                              p0(R)                     p1(p0(R))            p2(p1(p0(R)))
        //           | hopefully equal                                                  set equal  |
        // backward: p-0(p-1(p-2(p2(p1(p0(R))))))   p-1(p-2(p2(p1(p0(R)))))   p-2(p2(p1(p0(R))))   p2(p1(p0(R)))

        auto t1 = chrono::high_resolution_clock::now();

        for (size_t i = 0; i < INPUT_HYPERVECTOR_COUNT; ++i) {
            forward[i][0] = bhv::rand(); // TODO this and the eq should be outside of the timing

            for (size_t j = 0; j < different_permutations; ++j)
                forward[i][j + 1] = bhv::permute(forward[i][j], perms[j]);

            backward[i][different_permutations] = forward[i][different_permutations];

            for (size_t j = different_permutations; j > 0; --j)
                backward[i][j - 1] = bhv::permute(backward[i][j], -perms[j - 1]);

            correct &= bhv::eq(forward[i][0], backward[i][0]);
        }

        auto t2 = chrono::high_resolution_clock::now();
        total_test_time = (double) chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count();

        for (size_t i = 0; i < INPUT_HYPERVECTOR_COUNT; ++i) {
            for (size_t j = 0; j < different_permutations; ++j) free(forward[i][j]);
            for (size_t j = 0; j < different_permutations - 1; ++j) free(backward[i][j]); // -1, don't double free
        }
    } else {
//        word_t *running = bhv::rand();
        assert(false); // TODO
    }

    double mean_test_time = total_test_time / (double) test_count;
    if (display)
        cout << "correctly inverted: " << (correct ? "v" : "x") << ", in_cache: "
             << keep_in_cache << ", total: " << mean_test_time / 1000.0 << "µs" << endl;

    return mean_test_time;
}

float active_benchmark(bool display) {
    const int test_count = INPUT_HYPERVECTOR_COUNT;

    word_t *hvs [test_count];

    for (size_t i = 0; i < test_count; ++i) {
        hvs[i] = bhv::random((float)i/(float)test_count);
    }

    uint32_t observed_active[test_count];

    auto t1 = chrono::high_resolution_clock::now();
    for (size_t i = 0; i < test_count; ++i) {
        word_t *m = hvs[i];
        observed_active[i] = bhv::active(m);
        // observed_active[i] = bhv::active_avx512(m);
        // observed_active[i] = bhv::active_adder_avx2(m);
        // observed_active[i] = bhv::active_reference(m);
    }
    auto t2 = chrono::high_resolution_clock::now();

    bool correct = true;
    for (size_t i = 0; i < test_count; ++i) {
        correct &= observed_active[i] == bhv::active_reference(hvs[i]);
    }

    float mean_test_time = (float) chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count() / (float) test_count;
    if (display)
        cout << "correct " << (correct ? "v" : "x") << ", total: " << mean_test_time / 1000.0 << "µs" << endl;

    return mean_test_time;
}

float hamming_benchmark(bool display) {
    const int test_count = INPUT_HYPERVECTOR_COUNT/2;

    word_t *as [test_count];
    word_t *bs [test_count];

    for (size_t i = 0; i < test_count; ++i) {
        as[i] = bhv::random((float)i/(float)test_count);
        bs[i] = bhv::random((float)(test_count - i)/(float)test_count);
    }

    uint32_t observed_distance[test_count];

    auto t1 = chrono::high_resolution_clock::now();
    for (size_t i = 0; i < test_count; ++i) {
        observed_distance[i] = bhv::hamming(as[i], bs[i]);
        // observed_distance[i] = bhv::hamming_reference(as[i], bs[i]);
        // observed_distance[i] = bhv::hamming_adder_avx2(as[i], bs[i]);
        // observed_distance[i] = bhv::hamming_avx512(as[i], bs[i]);
    }
    auto t2 = chrono::high_resolution_clock::now();

    bool correct = true;
    for (size_t i = 0; i < test_count; ++i) {
        correct &= observed_distance[i] == bhv::hamming_reference(as[i], bs[i]);
    }

    float mean_test_time = (float) chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count() / (float) test_count;
    if (display)
        cout << "correct " << (correct ? "v" : "x") << ", total: " << mean_test_time / 1000.0 << "µs" << endl;

    return mean_test_time;
}

template <void F(word_t*, word_t*), void FC(word_t*, word_t*)>
float unary_benchmark(bool display,  bool keep_in_cache) {
    const int test_count = INPUT_HYPERVECTOR_COUNT;
    const int input_output_count = (keep_in_cache ? 1 : test_count);

    word_t *hvs [test_count];

    word_t *result_buffer = (word_t *) malloc(input_output_count * BYTES);

    for (size_t i = 0; i < test_count; ++i)
        hvs[i] = bhv::random((float)i/(float)test_count);

    volatile word_t something = 0;
    volatile word_t something_else = 0;

    auto t1 = chrono::high_resolution_clock::now();
    for (size_t i = 0; i < test_count; ++i) {
        const size_t io_buf_idx = (keep_in_cache ? 0 : i);

        word_t *m = result_buffer + (io_buf_idx * BYTES / sizeof(word_t));

        F(hvs[i], m);

        something = something ^ m[0] + 3 * m[4] + 5 * m[WORDS / 2] + 7 * m[WORDS - 1];
    }
    auto t2 = chrono::high_resolution_clock::now();

    for (size_t i = 0; i < test_count; ++i) {
        word_t *m = bhv::empty();

        FC(hvs[i], m);

        something_else = something_else ^ m[0] + 3 * m[4] + 5 * m[WORDS / 2] + 7 * m[WORDS - 1];
    }

    float mean_test_time = (float) chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count() / (float) test_count;
    if (display)
        cout << "equiv " << ((something == something_else) ? "v" : "x") << ", total: " << mean_test_time / 1000.0 << "µs" << endl;

    return mean_test_time;
}

template <void F(word_t*, word_t*, word_t*), void FC(word_t*, word_t*, word_t*)>
float binary_benchmark(bool display,  bool keep_in_cache) {
    const int test_count = INPUT_HYPERVECTOR_COUNT;
    const int input_output_count = (keep_in_cache ? 1 : test_count);

    word_t *hvs0 [test_count];
    word_t *hvs1 [test_count];

    word_t *result_buffer = (word_t *) malloc(input_output_count * BYTES);

    for (size_t i = 0; i < test_count; ++i)
        hvs0[i] = bhv::random((float)i/(float)test_count);

    memcpy(hvs1, hvs0, test_count * sizeof(word_t *));
    std::shuffle(hvs1, hvs1 + test_count, bhv::rng);

    volatile word_t something = 0;
    volatile word_t something_else = 0;

    auto t1 = chrono::high_resolution_clock::now();
    for (size_t i = 0; i < test_count; ++i) {
        const size_t io_buf_idx = (keep_in_cache ? 0 : i);

        word_t *m = result_buffer + (io_buf_idx * BYTES / sizeof(word_t));

        F(hvs0[i], hvs1[i], m);

        something = something ^ m[0] + 3 * m[4] + 5 * m[WORDS / 2] + 7 * m[WORDS - 1];
    }
    auto t2 = chrono::high_resolution_clock::now();

    for (size_t i = 0; i < test_count; ++i) {
        word_t *m = bhv::empty();

        FC(hvs0[i], hvs1[i], m);

        something_else = something_else ^ m[0] + 3 * m[4] + 5 * m[WORDS / 2] + 7 * m[WORDS - 1];
    }

    float mean_test_time = (float) chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count() / (float) test_count;
    if (display)
        cout << "equiv " << ((something == something_else) ? "v" : "x") << ", total: " << mean_test_time / 1000.0 << "µs" << endl;

    return mean_test_time;
}

typedef void(*TernaryFunc)(word_t*, word_t*, word_t*, word_t*);

template <TernaryFunc F, TernaryFunc FC>
float ternary_benchmark(bool display,  bool keep_in_cache) {
    const int test_count = INPUT_HYPERVECTOR_COUNT;
    const int input_output_count = (keep_in_cache ? 1 : test_count);

    word_t *hvs0 [test_count];
    word_t *hvs1 [test_count];
    word_t *hvs2 [test_count];

    word_t *result_buffer = (word_t *) malloc(input_output_count * BYTES);

    for (size_t i = 0; i < test_count; ++i)
        hvs0[i] = bhv::random((float)i/(float)test_count);

    memcpy(hvs1, hvs0, test_count * sizeof(word_t *));
    std::shuffle(hvs1, hvs1 + test_count, bhv::rng);
    memcpy(hvs2, hvs0, test_count * sizeof(word_t *));
    std::shuffle(hvs2, hvs2 + test_count, bhv::rng);

    volatile word_t something = 0;
    volatile word_t something_else = 0;

    auto t1 = chrono::high_resolution_clock::now();
    for (size_t i = 0; i < test_count; ++i) {
        const size_t io_buf_idx = (keep_in_cache ? 0 : i);

        word_t *m = result_buffer + (io_buf_idx * BYTES / sizeof(word_t));

        F(hvs0[i], hvs1[i], hvs2[i], m);

        something = something ^ m[0] + 3 * m[4] + 5 * m[WORDS / 2] + 7 * m[WORDS - 1];
    }
    auto t2 = chrono::high_resolution_clock::now();

    for (size_t i = 0; i < test_count; ++i) {
        word_t *m = bhv::empty();

        FC(hvs0[i], hvs1[i], hvs2[i], m);

        something_else = something_else ^ m[0] + 3 * m[4] + 5 * m[WORDS / 2] + 7 * m[WORDS - 1];
    }

    float mean_test_time = (float) chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count() / (float) test_count;
    if (display)
        cout << "equiv " << ((something == something_else) ? "v" : "x") << ", total: " << mean_test_time / 1000.0 << "µs" << endl;

    return mean_test_time;
}

inline void simulated_select(word_t *x, word_t *y, word_t *z, word_t *target) {
    bhv::dynamic_ternary_into_reference(x, y, z, target, 0xca);
};

inline void simulated_maj3(word_t *x, word_t *y, word_t *z, word_t *target) {
    bhv::dynamic_ternary_into_reference(x, y, z, target, 0xe8);
};

inline void simulated_any(word_t *x, word_t *y, word_t *z, word_t *target) {
    bhv::dynamic_ternary_into_reference(x, y, z, target, 0b11111110);
};

void __attribute__ ((noinline)) any_via_threshold(word_t *x, word_t *y, word_t *z, word_t *target) {
    word_t *xs [3] = {x, y, z};
    bhv::threshold_into(xs, 3, 0, target);
};


int main() {
    cout << "*-= WARMUP =-*" << endl;
    // burn some cycles to get the OS's attention
    volatile uint64_t x = 0x7834d688d8827099ULL;
    for (size_t i = 0; i < 50000000; ++i)
        x = x + (x % 7);

    cout << "*-= STARTING (" << x << ") =-*" << endl;
#ifdef TERNARY

    ternary_benchmark<simulated_select, bhv::select_into_reference>(false, true);

    cout << "*-= TERNARY =-*" << endl;
    cout << "*-= IN CACHE TESTS =-*" << endl;
    ternary_benchmark<simulated_select, bhv::select_into_reference>(true, true);
    ternary_benchmark<simulated_maj3, bhv::majority3_into>(true, true);
    ternary_benchmark<simulated_any, any_via_threshold>(true, true);

    cout << "*-= OUT OF CACHE TESTS =-*" << endl;
    ternary_benchmark<simulated_select, bhv::select_into_reference>(true, false);
    ternary_benchmark<simulated_maj3, bhv::majority3_into>(true, false);
    ternary_benchmark<simulated_any, any_via_threshold>(true, false);
#endif
#ifdef MAJ3
    ternary_benchmark<bhv::majority3_into, bhv::majority3_into_reference>(false, true);

    cout << "*-= MAJ3 =-*" << endl;
    cout << "*-= IN CACHE TESTS =-*" << endl;
    ternary_benchmark<bhv::majority3_into, bhv::majority3_into_reference>(true, true);
    ternary_benchmark<bhv::majority3_into, bhv::majority3_into_reference>(true, true);
    ternary_benchmark<bhv::majority3_into, bhv::majority3_into_reference>(true, true);

    cout << "*-= OUT OF CACHE TESTS =-*" << endl;
    ternary_benchmark<bhv::majority3_into, bhv::majority3_into_reference>(true, false);
    ternary_benchmark<bhv::majority3_into, bhv::majority3_into_reference>(true, false);
    ternary_benchmark<bhv::majority3_into, bhv::majority3_into_reference>(true, false);
#endif
#ifdef SELECT
    ternary_benchmark<bhv::select_into, bhv::select_into_reference>(false, true);

    cout << "*-= SELECT =-*" << endl;
    cout << "*-= IN CACHE TESTS =-*" << endl;
    ternary_benchmark<bhv::select_into, bhv::select_into_reference>(true, true);
    ternary_benchmark<bhv::select_into, bhv::select_into_reference>(true, true);
    ternary_benchmark<bhv::select_into, bhv::select_into_reference>(true, true);

    cout << "*-= OUT OF CACHE TESTS =-*" << endl;
    ternary_benchmark<bhv::select_into, bhv::select_into_reference>(true, false);
    ternary_benchmark<bhv::select_into, bhv::select_into_reference>(true, false);
    ternary_benchmark<bhv::select_into, bhv::select_into_reference>(true, false);
#endif
#ifdef XOR
    binary_benchmark<bhv::xor_into, bhv::xor_into>(false, true);

    cout << "*-= XOR =-*" << endl;
    cout << "*-= IN CACHE TESTS =-*" << endl;
    binary_benchmark<bhv::xor_into, bhv::xor_into>(true, true);
    binary_benchmark<bhv::xor_into, bhv::xor_into>(true, true);
    binary_benchmark<bhv::xor_into, bhv::xor_into>(true, true);

    cout << "*-= OUT OF CACHE TESTS =-*" << endl;
    binary_benchmark<bhv::xor_into, bhv::xor_into>(true, false);
    binary_benchmark<bhv::xor_into, bhv::xor_into>(true, false);
    binary_benchmark<bhv::xor_into, bhv::xor_into>(true, false);
#endif
#ifdef XOR
    binary_benchmark<bhv::xor_into, bhv::xor_into>(false, true);

    cout << "*-= XOR =-*" << endl;
    cout << "*-= IN CACHE TESTS =-*" << endl;
    binary_benchmark<bhv::xor_into, bhv::xor_into>(true, true);
    binary_benchmark<bhv::xor_into, bhv::xor_into>(true, true);
    binary_benchmark<bhv::xor_into, bhv::xor_into>(true, true);

    cout << "*-= OUT OF CACHE TESTS =-*" << endl;
    binary_benchmark<bhv::xor_into, bhv::xor_into>(true, false);
    binary_benchmark<bhv::xor_into, bhv::xor_into>(true, false);
    binary_benchmark<bhv::xor_into, bhv::xor_into>(true, false);
#endif
#ifdef OR
    binary_benchmark<bhv::or_into, bhv::or_into>(false, true);

    cout << "*-= OR =-*" << endl;
    cout << "*-= IN CACHE TESTS =-*" << endl;
    binary_benchmark<bhv::or_into, bhv::or_into>(true, true);
    binary_benchmark<bhv::or_into, bhv::or_into>(true, true);
    binary_benchmark<bhv::or_into, bhv::or_into>(true, true);

    cout << "*-= OUT OF CACHE TESTS =-*" << endl;
    binary_benchmark<bhv::or_into, bhv::or_into>(true, false);
    binary_benchmark<bhv::or_into, bhv::or_into>(true, false);
    binary_benchmark<bhv::or_into, bhv::or_into>(true, false);
#endif
#ifdef AND
    binary_benchmark<bhv::and_into, bhv::and_into>(false, true);

    cout << "*-= AND =-*" << endl;
    cout << "*-= IN CACHE TESTS =-*" << endl;
    binary_benchmark<bhv::and_into, bhv::and_into>(true, true);
    binary_benchmark<bhv::and_into, bhv::and_into>(true, true);
    binary_benchmark<bhv::and_into, bhv::and_into>(true, true);

    cout << "*-= OUT OF CACHE TESTS =-*" << endl;
    binary_benchmark<bhv::and_into, bhv::and_into>(true, false);
    binary_benchmark<bhv::and_into, bhv::and_into>(true, false);
    binary_benchmark<bhv::and_into, bhv::and_into>(true, false);
#endif
#ifdef REHASH
    unary_benchmark<bhv::rehash_into, bhv::rehash_into>(false, true);

    cout << "*-= REHASH =-*" << endl;
    cout << "*-= IN CACHE TESTS =-*" << endl;
    unary_benchmark<bhv::rehash_into, bhv::rehash_into>(true, true);
    unary_benchmark<bhv::rehash_into, bhv::rehash_into>(true, true);
    unary_benchmark<bhv::rehash_into, bhv::rehash_into>(true, true);

    cout << "*-= OUT OF CACHE TESTS =-*" << endl;
    unary_benchmark<bhv::rehash_into, bhv::rehash_into>(true, false);
    unary_benchmark<bhv::rehash_into, bhv::rehash_into>(true, false);
    unary_benchmark<bhv::rehash_into, bhv::rehash_into>(true, false);
#endif
#ifdef SWAP_HALVES
    unary_benchmark<bhv::swap_halves_into, bhv::swap_halves_into>(false, true);

    cout << "*-= SWAP_HALVES =-*" << endl;
    cout << "*-= IN CACHE TESTS =-*" << endl;
    unary_benchmark<bhv::swap_halves_into, bhv::swap_halves_into>(true, true);
    unary_benchmark<bhv::swap_halves_into, bhv::swap_halves_into>(true, true);
    unary_benchmark<bhv::swap_halves_into, bhv::swap_halves_into>(true, true);

    cout << "*-= OUT OF CACHE TESTS =-*" << endl;
    unary_benchmark<bhv::swap_halves_into, bhv::swap_halves_into>(true, false);
    unary_benchmark<bhv::swap_halves_into, bhv::swap_halves_into>(true, false);
    unary_benchmark<bhv::swap_halves_into, bhv::swap_halves_into>(true, false);
#endif
#ifdef INVERT
    unary_benchmark<bhv::invert_into, bhv::invert_into>(false, true);

    cout << "*-= INVERT =-*" << endl;
    cout << "*-= IN CACHE TESTS =-*" << endl;
    unary_benchmark<bhv::invert_into, bhv::invert_into>(true, true);
    unary_benchmark<bhv::invert_into, bhv::invert_into>(true, true);
    unary_benchmark<bhv::invert_into, bhv::invert_into>(true, true);

    cout << "*-= OUT OF CACHE TESTS =-*" << endl;
    unary_benchmark<bhv::invert_into, bhv::invert_into>(true, false);
    unary_benchmark<bhv::invert_into, bhv::invert_into>(true, false);
    unary_benchmark<bhv::invert_into, bhv::invert_into>(true, false);
#endif
#ifdef HAMMING
    hamming_benchmark(false);

    cout << "*-= HAMMING =-*" << endl;
    hamming_benchmark(true);
    hamming_benchmark(true);
    hamming_benchmark(true);
#endif
#ifdef ACTIVE
    active_benchmark(false);

    cout << "*-= ACTIVE =-*" << endl;
    active_benchmark(true);
    active_benchmark(true);
    active_benchmark(true);
#endif
#ifdef PERMUTE
    permute_benchmark<true, 100>(false);

    cout << "*-= PERMUTE =-*" << endl;
    permute_benchmark<true, 100>(true);
    permute_benchmark<true, 100>(true);
    permute_benchmark<true, 100>(true);
#endif
#ifdef RAND
    rand_benchmark(false, false);

    cout << "*-= RAND =-*" << endl;
    cout << "*-= IN CACHE TESTS =-*" << endl;
    rand_benchmark(true, true);
    rand_benchmark(true, true);
    rand_benchmark(true, true);

    cout << "*-= OUT OF CACHE TESTS =-*" << endl;
    rand_benchmark(true, false);
    rand_benchmark(true, false);
    rand_benchmark(true, false);
#endif
#ifdef RAND2
    int8_t pws[33];
    for (int8_t i = -16; i < 17; ++i)
        pws[i + 16] = i;

    rand2_benchmark(false, false, 4);
    cout << "*-= RAND2 =-*" << endl;

    cout << "*-= IN CACHE TESTS =-*" << endl;
    for (float p : pws)
        rand2_benchmark(true, true, p);

    cout << "*-= OUT OF CACHE TESTS =-*" << endl;
    for (float p : pws)
        rand2_benchmark(true, false, p);
#endif
#ifdef RANDOM
    random_benchmark(false, false, .1);

    cout << "*-= RANDOM =-*" << endl;
    cout << "*-= COMMON =-*" << endl;
    float common[13] = {.001, .01, .04, .2, .26, .48, .5, .52, .74,.8, .95, .99, .999};
    cout << "*-= IN CACHE TESTS =-*" << endl;
    double total = 0.;
    for (float p: common)
        total += random_benchmark(true, true, p);
    cout << "total: " << total << endl;
    total = 0;
    cout << "*-= OUT OF CACHE TESTS =-*" << endl;
    for (float p: common)
        total += random_benchmark(true, true, p);
    cout << "total: " << total << endl;

    cout << "*-= SMALL =-*" << endl;
    float small[9] = {1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2};
    cout << "*-= IN CACHE TESTS =-*" << endl;
    total = 0.;
    for (float p: small)
        total += random_benchmark(true, true, p);
    cout << "total: " << total << endl;
    total = 0;
    cout << "*-= OUT OF CACHE TESTS =-*" << endl;
    for (float p: small)
        total += random_benchmark(true, false, p);
    cout << "total: " << total << endl;

    cout << "*-= PERCENTAGES =-*" << endl;
    float perc[99];
    for (size_t i = 1; i < 100; ++i)
        perc[i - 1] = (float) i / 100.f;
    cout << "*-= IN CACHE TESTS =-*" << endl;
    total = 0.;
    for (float p: perc)
        total += random_benchmark(true, false, p);
    cout << "total: " << total << endl;
    total = 0;
    cout << "*-= OUT OF CACHE TESTS =-*" << endl;
    for (float p: perc)
        total += random_benchmark(true, false, p);
    cout << "total: " << total << endl;
#endif
#ifdef MAJ
    //Run one throw-away test to make sure the OS is ready to give us full resource
    majority_benchmark(3, false, false);

    cout << "*-= IN CACHE TESTS =-*" << endl;
    majority_benchmark(3, true, true);
    majority_benchmark(5, true, true);
    majority_benchmark(7, true, true);
    majority_benchmark(9, true, true);
    majority_benchmark(11, true, true);
    majority_benchmark(15, true, true);
    majority_benchmark(17, true, true);
    majority_benchmark(19, true, true);
    majority_benchmark(21, true, true);
    majority_benchmark(23, true, true);
    majority_benchmark(25, true, true);
    majority_benchmark(27, true, true);
    majority_benchmark(39, true, true);
    majority_benchmark(47, true, true);
    majority_benchmark(55, true, true);
    majority_benchmark(63, true, true);
    majority_benchmark(73, true, true);
    majority_benchmark(77, true, true);
    majority_benchmark(79, true, true);
    majority_benchmark(81, true, true);
    majority_benchmark(85, true, true);
    majority_benchmark(89, true, true);
    majority_benchmark(91, true, true);
    majority_benchmark(109, true, true);
    majority_benchmark(175, true, true);
    majority_benchmark(201, true, true);
    majority_benchmark(255, true, true);

    majority_benchmark(257, true, true);
    majority_benchmark(385, true, true);
    majority_benchmark(511, true, true);
    majority_benchmark(667, true, true);
    majority_benchmark(881, true, true);
    majority_benchmark(945, true, true);
    majority_benchmark(1021, true, true);
    majority_benchmark(2001, true, true);
    majority_benchmark(5001, true, true);
    majority_benchmark(9999, true, true);
    majority_benchmark(10003, true, true);
    majority_benchmark(20001, true, true);
    majority_benchmark(200001, true, true);
    majority_benchmark(1000001, true, true);

    cout << "*-= OUT OF CACHE TESTS =-*" << endl;
    majority_benchmark(3, true, false);
    majority_benchmark(5, true, false);
    majority_benchmark(7, true, false);
    majority_benchmark(9, true, false);
    majority_benchmark(11, true, false);
    majority_benchmark(27, true, false);
    majority_benchmark(39, true, false);
    majority_benchmark(47, true, false);
    majority_benchmark(55, true, false);
    majority_benchmark(63, true, false);
    majority_benchmark(73, true, false);
    majority_benchmark(77, true, false);
    majority_benchmark(79, true, false);
    majority_benchmark(81, true, false);
    majority_benchmark(85, true, false);
    majority_benchmark(89, true, false);
    majority_benchmark(91, true, false);
    majority_benchmark(109, true, false);
    majority_benchmark(175, true, false);
    majority_benchmark(201, true, false);
    majority_benchmark(255, true, false);
    majority_benchmark(257, true, false);
    majority_benchmark(313, true, false);
    majority_benchmark(385, true, false);
    majority_benchmark(511, true, false);
    majority_benchmark(667, true, false);
    majority_benchmark(881, true, false);
    majority_benchmark(945, true, false);
    majority_benchmark(1021, true, false);
    majority_benchmark(2001, true, false);
    majority_benchmark(5001, true, false);
    majority_benchmark(9999, true, false);
    majority_benchmark(10003, true, false);
    majority_benchmark(20001, true, false);
    majority_benchmark(200001, true, false);
    majority_benchmark(1000001, true, false);
#endif
#ifdef THRESHOLD
    threshold_benchmark(3, 0, .5, false, false);

    cout << "*-= THRESHOLD =-*" << endl;
    cout << "*-= IN CACHE TESTS =-*" << endl;
    threshold_benchmark(3, 0, .5, true, true);
    threshold_benchmark(10, 2, .3, true, true);
    threshold_benchmark(30, 20, .7, true, true);


    threshold_benchmark(100, 48, .5, true, true);
    threshold_benchmark(200, 50, .25, true, true);

    threshold_benchmark(3000, 1502, .5, true, true);
    threshold_benchmark(4000, 1000, .25, true, true);

    threshold_benchmark(200001, 0, .5, true, true);
    threshold_benchmark(200001, 200000, .5, true, true);

    threshold_benchmark(1000001, 498384, .5, true, true);

    cout << "*-= OUT OF CACHE TESTS =-*" << endl;
    threshold_benchmark(3, 0, .5, true, false);
    threshold_benchmark(10, 2, .3, true, false);
    threshold_benchmark(30, 20, .7, true, false);


    threshold_benchmark(100, 48, .5, true, false);
    threshold_benchmark(200, 50, .25, true, false);

    threshold_benchmark(3000, 1502, .5, true, false);
    threshold_benchmark(4000, 1000, .25, true, false);

    threshold_benchmark(200001, 0, .5, true, false);
    threshold_benchmark(200001, 200000, .5, true, false);

    threshold_benchmark(1000001, 498384, .5, true, false);
#endif
}
