#include <chrono>
#include <iostream>
#include <random>
#include <bitset>
#include <functional>
#include "core.h"

using namespace std;


void print_bits(int64_t w) {
    bitset<64> x(w);
    cout << x << endl;
}


void print_byte(int8_t w) {
    bitset<8> x(w);
    cout << x << endl;
}


void test_single_byte_permutation() {
    mt19937_64 rng;

    uint64_t r = rng();

    uint8_t r0 = 0b11110001;

    uint64_t id = (uint64_t) _mm_set_pi8(7, 6, 5, 4, 3, 2, 1, 0);

    uint64_t p1 = (uint64_t) _mm_set_pi8(7, 5, 3, 1, 6, 4, 2, 0);
    uint64_t p1_inv = (uint64_t) _mm_set_pi8(7, 2, 6, 2, 5, 1, 4, 0);

    uint64_t p2 = (uint64_t) _mm_set_pi8(7, 6, 5, 4, 0, 1, 2, 3);
    uint64_t p2_inv = p2;

    uint64_t rp1 = bhv::rand_byte_bits_permutation(5);
    uint64_t rp1_inv = bhv::byte_bits_permutation_invert(rp1);


    print_bits(rp1);
    print_bits(rp1_inv);

    print_byte(r0);

    uint8_t res = bhv::permute_single_byte_bits(r0, rp1);
    print_byte(res);

    uint8_t res_ = bhv::permute_single_byte_bits(res, rp1_inv);
    print_byte(res_);
}

void test_instruction_upto() {
    uint8_t to = 0;
    float_t remaining = 0;
//    uint64_t instruction = bhv::instruction_upto(.0000001, &to, &remaining);  //
    uint64_t instruction = bhv::instruction_upto(.5, &to, &remaining);  //
//    uint64_t instruction = bhv::instruction_upto(.625, &to, &remaining); // 10
//    uint64_t instruction = bhv::instruction_upto(.375, &to, &remaining);  // 01
//    uint64_t instruction = bhv::instruction_upto(.9, &to, &remaining, .005);  // 1110
//    uint64_t instruction = bhv::instruction_upto(.5625, &to, &remaining);  // 100
//    uint64_t instruction = bhv::instruction_upto(123.f/256.f, &to, &remaining);  // 0111101
//    uint64_t instruction = bhv::instruction_upto(.5625001, &to, &remaining);  // 100
//    uint64_t instruction = bhv::instruction_upto(.5624999, &to, &remaining);  // 100

    cout << "to: " << (uint32_t) to << endl;
    cout << "rem: " << remaining << endl;
    print_bits(instruction);

    for (uint8_t i = to - 1; i < to; --i)
        cout << ((instruction & (1 << i)) >> i);

    cout << endl;

    word_t *x = bhv::empty();
    bhv::random_into_tree_sparse_avx2(x, .5);
    cout << "active: " << (double) bhv::active(x) / (double) BITS << endl;
}

void test_ternary_instruction() {
    uint8_t buffer [24];
    uint8_t to = 0;
//    int8_t finalizer = bhv::ternary_instruction(.0000001, buffer, &to, 1e-3);
//    int8_t finalizer = bhv::ternary_instruction(.5, buffer, &to);
//    buffer: 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000
//    to: 0
//    finalizer: -1

//    int8_t finalizer = bhv::ternary_instruction(.25, buffer, &to);
//    buffer: 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000
//    to: 0
//    finalizer: -1

//    int8_t finalizer = bhv::ternary_instruction(.625, buffer, &to);
//    buffer: 11111000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000
//    to: 1
//    finalizer: -1

//    int8_t finalizer = bhv::ternary_instruction(.375, buffer, &to, 1e-3);
//    buffer: 11100000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000
//    to: 1
//    finalizer: -1

//    int8_t finalizer = bhv::ternary_instruction(.9, buffer, &to, .005);  // 1110
//    buffer: 11111110 11111000 11100000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000
//    to: 3
//    finalizer: -1

//    uint64_t instruction = bhv::instruction_upto(.5625, &to, &remaining);  // 100

    int8_t finalizer = bhv::ternary_instruction(123.f/256.f, buffer, &to);  // 0111101
//    buffer: 11100000 11111110 11111000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000
//    to: 3
//    finalizer: 1

//    uint64_t instruction = bhv::instruction_upto(.5625001, &to, &remaining);  // 100
//    uint64_t instruction = bhv::instruction_upto(.5624999, &to, &remaining);  // 100

    cout << "buffer: ";
    for (uint8_t b : buffer)
        cout << bitset<8>(b) << " ";
    cout << endl;
    cout << "to: " << (uint32_t) to << endl;
    cout << "finalizer: " << (int32_t) finalizer << endl;

//    word_t *x = bhv::empty();
//    bhv::random_into_ternary_tree_avx512(x, .5);
//    cout << "active: " << (double) bhv::active(x) / (double) BITS << endl;

//    bhv::random_into_ternary_tree_avx512(x, 123.f/256.f);
//    cout << "expected: " << 123.f/256.f << ", active: " << (double) bhv::active(x) / (double) BITS << endl;
}

void test_independent_opt() {
    auto v_ref = bhv::rand();
    auto v = bhv::rand();

    cout << bhv::hamming(v_ref, v) << " |v_ref,initial|" << endl;

    auto close_to_ref = [v_ref](word_t *x) { return bhv::hamming(x, v_ref); };
    bhv::opt_independent<uint64_t>(v, close_to_ref);

    cout << bhv::hamming(v_ref, v) << " |v_ref,final|" << endl;

    auto full = bhv::one();

    auto half = [](word_t *x) { return abs(int64_t(bhv::active(x) - BITS/2)); };
    bhv::opt_independent<uint64_t>(full, half);

    cout << bhv::active(full) << " |full_opt|  (" << BITS/2 << ")" << endl;
}

void test_easy_search() {
    auto v_ref = bhv::rand();
    auto v = bhv::rand();

    cout << bhv::hamming(v_ref, v) << " |v_ref,initial|" << endl;

    auto close_to_ref = [v_ref](word_t *x) { return bhv::hamming(x, v_ref); };
    bhv::opt_linear_search<uint64_t>(v, close_to_ref);

    cout << bhv::hamming(v_ref, v) << " |v_ref,final|" << endl;
}

void test_harder_search() {
    auto v_ref = bhv::rand();
    auto v = bhv::rand();

    auto close_to_ref = [v_ref](word_t *x) { word_t tmp [WORDS]; bhv::roll_word_bits_into(x, 12, tmp); bhv::xor_into(x, tmp, tmp); return bhv::hamming(tmp, v_ref); };
    auto t1 = chrono::high_resolution_clock::now();

    bhv::opt_linear_search<uint64_t>(v, close_to_ref, 100000000);

    auto t2 = chrono::high_resolution_clock::now();

    cout << close_to_ref(v) << " loss" << endl;
    double test_time = (double) chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count() / (double) (1000000000);
    cout << test_time << " seconds" << endl;
}

void test_opt_centroid() {
    // while the min_sum vector is unique and equals the MAJ
    // there is a family of vectors min_diff vectors unrelated to each other
    auto v = bhv::rand();
    auto x = bhv::rand();
    auto y = bhv::rand();
    auto z = bhv::rand();
    word_t m [WORDS];
    bhv::majority3_into(x, y, z, m);

    auto min_sum = [x, y, z](word_t *a) { auto dx = bhv::hamming(a, x); auto dy = bhv::hamming(a, y); auto dz = bhv::hamming(a, z); return dx + dy + dz; };
//    auto min_diff = [x, y, z](word_t *a) { int64_t dx = bhv::hamming(a, x); int64_t dy = bhv::hamming(a, y); int64_t dz = bhv::hamming(a, z); return abs(dx - dy) + abs(dy - dz) + abs(dx - dz); };

    auto t1 = chrono::high_resolution_clock::now();
    bhv::opt_linear_search<uint64_t>(v, min_sum, 1000000);
    auto t2 = chrono::high_resolution_clock::now();

    cout << min_sum(v) << " loss" << endl;
    cout << bhv::hamming(v, m) << " |maj(x,y,z),final|" << endl;
    cout << bhv::hamming(v, x) << " |x,final|, " << bhv::hamming(v, y) << " |y,final|, " << bhv::hamming(v, z) << " |z,final|" << endl;
    cout << bhv::hash(v) << " h" << endl;
    double test_time = (double) chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count() / (double) (1000000000);
    cout << test_time << " seconds" << endl;
}

int main() {
    test_opt_centroid();
}