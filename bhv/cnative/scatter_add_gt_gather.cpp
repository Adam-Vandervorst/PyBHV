#include <iostream>
#include <chrono>
#include <bitset>
#include <immintrin.h>

using namespace std;

void print_bits(int64_t w) {
    std::bitset<64> x(w);
    std::cout << x << std::endl;
}

int main() {
    // the data we care about
    uint64_t b1 = 0b01101111;
    uint64_t b2 = 0b00010011;
    uint64_t b3 = 0b00110110;
    uint64_t re = 0b00110111; // expected result of MAJ3

    // scatter so each bit of the byte lines up with a byte of a word
    uint64_t b1_byte_spread = _pdep_u64(b1, 0x0101010101010101);
    print_bits(b1_byte_spread);
    uint64_t b2_byte_spread = _pdep_u64(b2, 0x0101010101010101);
    print_bits(b2_byte_spread);
    uint64_t b3_byte_spread = _pdep_u64(b3, 0x0101010101010101);
    print_bits(b3_byte_spread);

    // embedding it into __m128 so we can do bytewise processing
    __m128i b1_simd = _mm_setr_epi64(_mm_cvtsi64_m64(0), _mm_cvtsi64_m64(b1_byte_spread));
    __m128i b2_simd = _mm_setr_epi64(_mm_cvtsi64_m64(0), _mm_cvtsi64_m64(b2_byte_spread));
    __m128i b3_simd = _mm_setr_epi64(_mm_cvtsi64_m64(0), _mm_cvtsi64_m64(b3_byte_spread));

    // bytewise addition
    __m128i counts = _mm_add_epi8(_mm_add_epi8(b1_simd, b2_simd), b3_simd);
    print_bits(_mm_extract_epi64(counts, 1));

    // out threshold, in this case hardcoded for 3 inputs
    __m128i threshold = _mm_setr_epi64(_mm_cvtsi64_m64(0), _mm_cvtsi64_m64(0x0101010101010101));

    // bytewise greater then threshold
    uint64_t maj_word = _mm_extract_epi64(_mm_cmpgt_epi8(counts, threshold), 1);
    print_bits(maj_word);

    // gather so each gt result is a single bit
    uint64_t maj = _pext_u64(maj_word, 0x0101010101010101);

    // print whether the result is as exptected
    std::cout << (re == maj) << std::endl;
    return 0;
}
