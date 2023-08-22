#include <iostream>
#include <chrono>

#include "core.h"

using namespace std;


#define MAJ_INPUT_COUNT 2000001


float majority_benchmark(size_t n, bool display, bool keep_in_cache) {
    //For the simple cases, like 3 vectors, we want a lot of tests to get a reliable number
    //but allocating 2,000 vectors * 10,000 tests starts to exceed resident memory and we end
    //up paying disk swap penalties.  Therefore we do fewer tests in the case with more hypervectors
    const size_t test_count = MAJ_INPUT_COUNT / n;
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

    auto t1 = chrono::high_resolution_clock::now();
    for (size_t i = 0; i < test_count; i++) {
        const size_t io_buf_idx = (keep_in_cache ? 0 : i);

        word_t *m = result_buffer + (io_buf_idx * BYTES / sizeof(word_t));
        word_t **rs = inputs[io_buf_idx];

        bhv::true_majority_into(rs, n, m);

        // So the test operation doesn't get optimized away
        something ^= m[0] + 3 * m[4] + 5 * m[WORDS / 2] + 7 * m[WORDS - 1];
    }
    auto t2 = chrono::high_resolution_clock::now();

    double mean_test_time = (double) chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count() / (double) test_count;
    if (display)
        cout << n << "," << mean_test_time / (double) n << endl;

    for (size_t i = 0; i < input_output_count; i++) {
        word_t **rs = inputs[i];
        for (size_t j = 0; j < n; ++j) free(rs[j]);
        free(rs);
    }
    free(result_buffer);
    free(inputs);

    return mean_test_time;
}


int main() {
    // burn some cycles to get the OS's attention
    volatile uint64_t x = 0x7834d688d8827099ULL;
    for (size_t i = 0; i < 50000000; ++i)
        x = x + (x % 7);

    majority_benchmark(3, false, false);

    for (size_t i = 1; i < 200; i += 2) majority_benchmark(i, true, true);
    for (size_t i = 201; i < 2000; i += 20) majority_benchmark(i, true, true);
    for (size_t i = 2001; i < 20000; i += 200) majority_benchmark(i, true, true);
    for (size_t i = 20001; i < 200000; i += 2000) majority_benchmark(i, true, true);
    for (size_t i = 200001; i <= 2000001; i += 20000) majority_benchmark(i, true, true);

    cout << endl;

    for (size_t i = 1; i < 200; i += 2) majority_benchmark(i, true, false);
    for (size_t i = 201; i < 2000; i += 20) majority_benchmark(i, true, false);
    for (size_t i = 2001; i < 20000; i += 200) majority_benchmark(i, true, false);
    for (size_t i = 20001; i < 200000; i += 2000) majority_benchmark(i, true, false);
    for (size_t i = 200001; i <= 2000001; i += 20000) majority_benchmark(i, true, false);
}
