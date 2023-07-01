#include <iostream>
#include <chrono>

#include "packed.h"

using namespace std;

#define INPUT_HYPERVECTOR_COUNT 1000000
float majority_benchmark(int n, bool display, bool keep_in_cache) {
    //For the simple cases, like 3 vectors, we want a lot of tests to get a reliable number
    //but allocating 2,000 vectors * 10,000 tests starts to exceed resident memory and we end
    //up paying disk swap penalties.  Therefore we do fewer tests in the case with more hypervectors
    const int test_count = INPUT_HYPERVECTOR_COUNT / n;
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

        bhv::byte_threshold_into(rs, n, n/2, m);

        // So the test operation doesn't get optimized away
        something = something ^ m[0] ^ m[4] ^ m[8] ^ m[12];
    }
    auto t2 = chrono::high_resolution_clock::now();

    float mean_test_time = (float)chrono::duration_cast<chrono::nanoseconds>(t2-t1).count() / (float)test_count;

    volatile word_t something_else = 0;

    for (int i=0; i<test_count; i++) {
        const int io_buf_idx = (keep_in_cache? 0 : i);

        word_t** rs = inputs[io_buf_idx];

        word_t* m = bhv::threshold(rs, n, n/2);
        // So the test operation doesn't get optimized away
        something_else = something_else ^ m[0] ^ m[4] ^ m[8] ^ m[12];
    }

    if (display)
        cout << n << " hypervectors, correct: " << ((something == something_else) ? "V" : "X") << ", in_cache: " << keep_in_cache << ", total: " << mean_test_time / 1000.0 << "µs, normalized: " << mean_test_time / (float)n << "ns/vec" << endl;

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

int main() {
    //Run one throw-away test to make sure the OS is ready to give us full resource
    majority_benchmark(3, true, false);

    cout << "*-= IN CACHE TESTS =-*" << endl;
//    majority_benchmark(3, true, true);
    majority_benchmark(5, true, true);
    majority_benchmark(9, true, true);
    majority_benchmark(27, true, true);
    majority_benchmark(81, true, true);
    majority_benchmark(201, true, true);
//    majority_benchmark(667, true, true);
//    majority_benchmark(2001, true, true);

    cout << "*-= OUT OF CACHE TESTS =-*" << endl;
//    majority_benchmark(3, true, false);
    majority_benchmark(5, true, false);
    majority_benchmark(9, true, false);
    majority_benchmark(27, true, false);
    majority_benchmark(81, true, false);
    majority_benchmark(201, true, false); // FIXME, byte_threshold_into fails on this case
//    majority_benchmark(667, true, false);
//    majority_benchmark(2001, true, false);
}