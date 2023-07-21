#include <iostream>
#include <random>
#include <bitset>
#include "packed.h"


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

    uint8_t r0 = 0b01110001;

    uint64_t id = (uint64_t)_mm_set_pi8(7, 6, 5, 4, 3, 2, 1, 0);

    uint64_t p1 = (uint64_t)_mm_set_pi8(7, 5, 3, 1, 6, 4, 2, 0);
    uint64_t p1_inv = (uint64_t)_mm_set_pi8(7, 2, 6, 2, 5, 1, 4, 0);

    uint64_t p2 = (uint64_t)_mm_set_pi8(7, 6, 5, 4, 0, 1, 2, 3);
    uint64_t p2_inv = p2;

    uint64_t rp1 = bhv::rand_bits_of_byte_permutation(2);
    uint64_t rp1_inv = bhv::bits_of_byte_permutation_invert(rp1);


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
//    uint64_t instruction = bhv::instruction_upto(.5, &to, &remaining);  //
//    uint64_t instruction = bhv::instruction_upto(.625, &to); // 10
//    uint64_t instruction = bhv::instruction_upto(.375, &to);  // 01
    uint64_t instruction = bhv::instruction_upto(.9, &to, &remaining, .005);  // 1110
//    uint64_t instruction = bhv::instruction_upto(.5625, &to, &remaining);  // 100
//        uint64_t instruction = bhv::instruction_upto(123.f/256.f, &to, &remaining);  // 0111101
//        uint64_t instruction = bhv::instruction_upto(.5625001, &to, &remaining);  // 100
//        uint64_t instruction = bhv::instruction_upto(.5624999, &to, &remaining);  // 100

    std::cout << "to: " << (uint32_t)to << std::endl;
    std::cout << "rem: " << remaining << std::endl;
    print_bits(instruction);

    for (uint8_t i = to - 1; i < to; --i)
        std::cout << ((instruction & (1 << i)) >> i);

    std::cout << std::endl;

    word_t* x = bhv::empty();
    bhv::random_into_tree_sparse(x, .9);
    std::cout << "active: " << (double)bhv::active(x)/(double )BITS << std::endl;
}


int main() {
    test_instruction_upto();
}