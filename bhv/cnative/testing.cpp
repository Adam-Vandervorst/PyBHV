#include <iostream>
#include <random>
#include <bitset>
#include "core.h"


void print_bits(int64_t w) {
    std::bitset<64> x(w);
    std::cout << x << std::endl;
}


void print_byte(int8_t w) {
    std::bitset<8> x(w);
    std::cout << x << std::endl;
}


void test_single_byte_permutation() {
    std::mt19937_64 rng;

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

    std::cout << "to: " << (uint32_t) to << std::endl;
    std::cout << "rem: " << remaining << std::endl;
    print_bits(instruction);

    for (uint8_t i = to - 1; i < to; --i)
        std::cout << ((instruction & (1 << i)) >> i);

    std::cout << std::endl;

    word_t *x = bhv::empty();
    bhv::random_into_tree_sparse_avx2(x, .5);
    std::cout << "active: " << (double) bhv::active(x) / (double) BITS << std::endl;
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

    std::cout << "buffer: ";
    for (uint8_t b : buffer)
        std::cout << std::bitset<8>(b) << " ";
    std::cout << std::endl;
    std::cout << "to: " << (uint32_t) to << std::endl;
    std::cout << "finalizer: " << (int32_t) finalizer << std::endl;

//    word_t *x = bhv::empty();
//    bhv::random_into_ternary_tree_avx512(x, .5);
//    std::cout << "active: " << (double) bhv::active(x) / (double) BITS << std::endl;

//    bhv::random_into_ternary_tree_avx512(x, 123.f/256.f);
//    std::cout << "expected: " << 123.f/256.f << ", active: " << (double) bhv::active(x) / (double) BITS << std::endl;
}


int main() {
    test_ternary_instruction();
}