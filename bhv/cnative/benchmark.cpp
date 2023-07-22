#include <iostream>
#include <chrono>

#include "packed.h"

using namespace std;

#define MAJ_INPUT_HYPERVECTOR_COUNT 1000000
#define INPUT_HYPERVECTOR_COUNT 100000


float majority_benchmark(int n, bool display, bool keep_in_cache) {
    //For the simple cases, like 3 vectors, we want a lot of tests to get a reliable number
    //but allocating 2,000 vectors * 10,000 tests starts to exceed resident memory and we end
    //up paying disk swap penalties.  Therefore we do fewer tests in the case with more hypervectors
    const int test_count = MAJ_INPUT_HYPERVECTOR_COUNT / n;
    const int input_output_count = (keep_in_cache? 1 : test_count);

    //Init n random vectors for each test
    word_t** inputs[input_output_count];
    for (int i=0; i<input_output_count; i++){
        word_t** rs = (word_t**)malloc(sizeof(word_t**) * n);
        for (size_t i = 0; i < n; ++i) {
            rs[i] = bhv::rand();
        }
        inputs[i] = rs;
    }

    //Allocate a buffer for TEST_COUNT results
    word_t* result_buffer = (word_t*)malloc(input_output_count * BYTES);

    // Gotta assign the result to a volatile, so the test operation doesn't get optimized away
    volatile word_t something = 0;

    auto t1 = chrono::high_resolution_clock::now();
    for (int i=0; i<test_count; i++) {
        const int io_buf_idx = (keep_in_cache? 0 : i);

        word_t* m = result_buffer + (io_buf_idx * BYTES / sizeof(word_t));
        word_t** rs = inputs[io_buf_idx];

        bhv::true_majority_into(rs, n, m);

        // So the test operation doesn't get optimized away
        something = something ^ m[0] + 3*m[4] + 5*m[WORDS/2] + 7*m[WORDS - 1];
    }
    auto t2 = chrono::high_resolution_clock::now();

    float mean_test_time = (float)chrono::duration_cast<chrono::nanoseconds>(t2-t1).count() / (float)test_count;
    if (display)
        cout << n << " hypervectors, in_cache: " << keep_in_cache << ", total: " << mean_test_time / 1000.0 << "µs, normalized: " << mean_test_time / (float)n << "ns/vec" << endl;

    //Clean up our mess
    for (int i=0; i<input_output_count; i++){
        word_t** rs = inputs[i];
        for (size_t i = 0; i < n; ++i) {
            free(rs[i]);
        }
        free(rs);
    }
    free(result_buffer);

    return mean_test_time;
}


float rand_benchmark(bool display, bool keep_in_cache) {
    const int test_count = INPUT_HYPERVECTOR_COUNT;
    const int input_output_count = (keep_in_cache? 1 : test_count);

    //Allocate a buffer for TEST_COUNT results
    word_t* result_buffer = (word_t*)malloc(input_output_count * BYTES);

    // Gotta assign the result to a volatile, so the test operation doesn't get optimized away
    volatile word_t something = 0;

    auto t1 = chrono::high_resolution_clock::now();
    for (int i=0; i<test_count; i++) {
        const int io_buf_idx = (keep_in_cache? 0 : i);

        word_t* m = result_buffer + (io_buf_idx * BYTES / sizeof(word_t));

        bhv::rand_into_avx2(m);
        something = something ^ m[0] + 3*m[4] + 5*m[WORDS/2] + 7*m[WORDS - 1];
    }
    auto t2 = chrono::high_resolution_clock::now();

    float mean_test_time = (float)chrono::duration_cast<chrono::nanoseconds>(t2-t1).count() / (float)test_count;
    if (display)
        cout << "in_cache: " << keep_in_cache << ", total: " << mean_test_time / 1000.0 << "µs" << endl;

    //Clean up our mess
    free(result_buffer);

    return mean_test_time;
}

float rand2_benchmark(bool display, bool keep_in_cache, int pow) {
    const int test_count = INPUT_HYPERVECTOR_COUNT;
    const int input_output_count = (keep_in_cache? 1 : test_count);
    volatile int p = pow;

    //Allocate a buffer for TEST_COUNT results
    word_t* result_buffer = (word_t*)malloc(input_output_count * BYTES);

    double observed_frac [test_count];

    auto t1 = chrono::high_resolution_clock::now();
    for (int i=0; i<test_count; i++) {
        const int io_buf_idx = (keep_in_cache? 0 : i);

        word_t* m = result_buffer + (io_buf_idx * BYTES / sizeof(word_t));

        bhv::rand2_into(m, p);

        // once runtime of random_into drops under 500ns, consider removing this
        observed_frac[i] = (double)bhv::active(m)/(double)BITS;
    }
    auto t2 = chrono::high_resolution_clock::now();

    float mean_test_time = (float)chrono::duration_cast<chrono::nanoseconds>(t2-t1).count() / (float)test_count;
    std::sort(observed_frac, observed_frac + test_count);
    double mean_observed = std::reduce(observed_frac, observed_frac + test_count, 0., std::plus<double>())/(double)test_count;
    mean_observed = mean_observed > .5 ? 1 + std::log2(1 - mean_observed) : -1 - std::log2(mean_observed);
    if (display)
        cout << pow << "pow, observed: " << mean_observed << ", in_cache: " << keep_in_cache << ", total: " << mean_test_time / 1000.0 << "µs" << endl;

    //Clean up our mess
    free(result_buffer);

    return mean_test_time;
}


float random_benchmark(bool display, bool keep_in_cache, float base_frac) {
    const int test_count = INPUT_HYPERVECTOR_COUNT;
    const int input_output_count = (keep_in_cache? 1 : test_count);
    volatile double n = ((double)(rand() - RAND_MAX/2)/(double)(RAND_MAX))/1000.;
    float p = base_frac + n;

    //Allocate a buffer for TEST_COUNT results
    word_t* result_buffer = (word_t*)malloc(input_output_count * BYTES);

    double observed_frac [test_count];

    auto t1 = chrono::high_resolution_clock::now();
    for (int i=0; i<test_count; i++) {
        const int io_buf_idx = (keep_in_cache? 0 : i);

        word_t* m = result_buffer + (io_buf_idx * BYTES / sizeof(word_t));

//        bhv::random_into(m, p); // baseline
//        bhv::random_into_tree_avx2(m, p); // full tree expansion
//        bhv::random_into_1tree_sparse(m, p); // 1 level of tree expansion into sparse
        bhv::random_into_tree_sparse(m, p); // full tree expansion + sparse + some short-circuiting

        // once runtime of random_into drops under 500ns, consider removing this
        observed_frac[i] = (double)bhv::active(m)/(double)BITS;
    }
    auto t2 = chrono::high_resolution_clock::now();

    float mean_test_time = (float)chrono::duration_cast<chrono::nanoseconds>(t2-t1).count() / (float)test_count;
    std::sort(observed_frac, observed_frac + test_count);
    double mean_observed_frac = std::reduce(observed_frac, observed_frac + test_count, 0., std::plus<double>())/(double)test_count - n;
    if (display)
        cout << base_frac << "frac, observed: " << abs(base_frac - mean_observed_frac) << ", in_cache: " << keep_in_cache << ", total: " << mean_test_time / 1000.0 << "µs" << endl;

    //Clean up our mess
    free(result_buffer);

    return mean_test_time;
}


//#define MAJ
//#define RAND
//#define RAND2
#define RANDOM
//#define PERMUTE


int main() {
#ifdef PERMUTE

#endif
#ifdef RAND
    rand_benchmark(false, false);

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

    cout << "*-= IN CACHE TESTS =-*" << endl;
    for (float p : pws)
        rand2_benchmark(true, true, p);

    cout << "*-= OUT OF CACHE TESTS =-*" << endl;
    for (float p : pws)
        rand2_benchmark(true, true, p);

#endif
#ifdef RANDOM
//    float ps[9] = {1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2};
//    float ps[13] = {.001, .01, .04, .2, .26, .48, .5, .52, .74,.8, .95, .99, .999};
    float ps[99];
    for (size_t i = 1; i < 100; ++i)
        ps[i - 1] = (float)i/100.f;

    random_benchmark(false, false, .1);

    cout << "*-= IN CACHE TESTS =-*" << endl;
    for (float p : ps)
        random_benchmark(true, true, p);

    cout << "*-= OUT OF CACHE TESTS =-*" << endl;
    for (float p : ps)
        random_benchmark(true, true, p);

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
    // majority_benchmark(175, true, true);
    // majority_benchmark(201, true, true);
    // majority_benchmark(255, true, true);
    // majority_benchmark(256, true, true);
    // majority_benchmark(385, true, true);
    // majority_benchmark(511, true, true);
    // majority_benchmark(667, true, true);
    // majority_benchmark(881, true, true);
    // majority_benchmark(945, true, true);
    // majority_benchmark(1021, true, true);
    // majority_benchmark(2001, true, true);
    // majority_benchmark(5001, true, true);
    // majority_benchmark(9999, true, true);
    // majority_benchmark(10003, true, true);
    // majority_benchmark(20001, true, true);
    // majority_benchmark(200001, true, true);

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
#endif
}