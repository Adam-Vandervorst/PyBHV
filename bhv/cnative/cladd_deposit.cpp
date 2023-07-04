#include <cstdint>
#include <cstdio>
#include <bitset>
#include <immintrin.h>
#include <iostream>

int main()
{
    uint64_t Bits = 0b11101111;
    // Print input value in binary
    std::cout <<  "bits:    " << std::bitset<64>(Bits) << std::endl;

    __m128i Result = _mm_clmulepi64_si128(
            _mm_set_epi64x(0, Bits),
            _mm_set_epi64x(0, Bits),
            0
    );

    std::cout << "crumbs:  " << std::bitset<64>(_mm_extract_epi64(Result, 1)) << std::bitset<64>(_mm_extract_epi64(Result, 0)) << std::endl;

    Result = _mm_clmulepi64_si128(
            Result,
            Result,
            0
    );
    std::cout << "nibbles: " << std::bitset<64>(_mm_extract_epi64(Result, 1)) << std::bitset<64>(_mm_extract_epi64(Result, 0)) << std::endl;

    Result = _mm_clmulepi64_si128(
            Result,
            Result,
            0
    );
    std::cout << "bytes:   " << std::bitset<64>(_mm_extract_epi64(Result, 1)) << std::bitset<64>(_mm_extract_epi64(Result, 0)) << std::endl;

    Result = _mm_clmulepi64_si128(
            Result,
            Result,
            0
    );
    std::cout << "shorts:  " << std::bitset<64>(_mm_extract_epi64(Result, 1)) << std::bitset<64>(_mm_extract_epi64(Result, 0)) << std::endl;

}