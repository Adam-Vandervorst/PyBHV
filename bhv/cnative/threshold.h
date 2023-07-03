
#include <immintrin.h>

#if __AVX512BW__
/// @brief AVX-512 implementation of threshold_into using a 2-Byte counter
/// @note TODO! For large numbers of N, performing the compare and write is inferior to
// separating them into two loops, in the same way that threshold_into_generic does
void threshold_into_short_avx512(word_t ** xs, uint16_t size, uint8_t threshold, word_t* dst) {
    __m512i threshold_simd = _mm512_set1_epi16(threshold);
    uint8_t** xs_bytes = (uint8_t**)xs;
    uint8_t* dst_bytes = (uint8_t*)dst;

    for (byte_iter_t byte_id = 0; byte_id < BYTES; byte_id += 4) {
        __m512i total_simd = _mm512_set1_epi16(0);

        for (u_int16_t i = 0; i < size; ++i) {
            uint8_t* bytes_i = xs_bytes[i];
            uint64_t spread_words[4] = {
                _pdep_u64(bytes_i[byte_id], 0x0001000100010001),
                _pdep_u64(bytes_i[byte_id + 1], 0x0001000100010001),
                _pdep_u64(bytes_i[byte_id + 2], 0x0001000100010001),
                _pdep_u64(bytes_i[byte_id + 3], 0x0001000100010001),
            };

            total_simd = _mm512_add_epi16(total_simd, *((__m512i *)spread_words));
        }

        __mmask32 maj_bits = _mm512_cmpgt_epu16_mask(total_simd, threshold_simd);
        *((uint32_t*)&(dst_bytes[byte_id])) = maj_bits;
    }
}
#endif //__AVX512BW__

/// @brief AVX-256 implementation of threshold_into using a 2-Byte counter
/// @note The AVX-256 implementation is faster than threshold_into_short_avx256 for smaller
/// values or N, even when AVX-512  is available, probably on account of less cache pressure.
void threshold_into_short_avx256(word_t ** xs, uint16_t size, uint8_t threshold, word_t* dst) {
    __m256i threshold_simd = _mm256_set1_epi16(threshold);
    uint8_t** xs_bytes = (uint8_t**)xs;
    uint8_t* dst_bytes = (uint8_t*)dst;

    for (byte_iter_t byte_id = 0; byte_id < BYTES; byte_id += 2) {
        __m256i total_simd = _mm256_set1_epi16(0);

        for (uint16_t i = 0; i < size; ++i) {
            uint8_t* bytes_i = xs_bytes[i];
            uint64_t spread_words[2] = {
                _pdep_u64(bytes_i[byte_id], 0x0001000100010001),
                _pdep_u64(bytes_i[byte_id + 1], 0x0001000100010001),
            };

            total_simd = _mm256_add_epi16(total_simd, *((__m256i *)spread_words));
        }

        uint64_t maj_words[4];
        *(__m256i *) maj_words = _mm256_cmpgt_epi16(total_simd, threshold_simd);

        dst_bytes[byte_id] = (uint8_t)_pext_u64(maj_words[0], 0x0001000100010001);
        dst_bytes[byte_id + 1] = (uint8_t)_pext_u64(maj_words[1], 0x0001000100010001);
    }
}

#if __AVX512BW__
/// @brief AVX-512 implementation of threshold_into using a 1-Byte counter
/// @note Experimentally, a native AVX-512 implementation isn't faster because it tends
///  to stall waiting on the pdep instructions.  But there is some benefit to the 
///  _mm256_cmpgt_epu8_mask instruction.
void threshold_into_byte_avx512(word_t ** xs, uint8_t size, uint8_t threshold, word_t* dst) {
    __m256i threshold_simd = _mm256_set1_epi8(threshold);
    uint8_t** xs_bytes = (uint8_t**)xs;
    uint8_t* dst_bytes = (uint8_t*)dst;

    for (byte_iter_t byte_id = 0; byte_id < BYTES; byte_id += 4) {
        __m256i total_simd = _mm256_set1_epi8(0);

        for (u_int8_t i = 0; i < size; ++i) {
            uint8_t* bytes_i = xs_bytes[i];
            uint64_t spread_words[4] = {
                _pdep_u64(bytes_i[byte_id], 0x0101010101010101),
                _pdep_u64(bytes_i[byte_id + 1], 0x0101010101010101),
                _pdep_u64(bytes_i[byte_id + 2], 0x0101010101010101),
                _pdep_u64(bytes_i[byte_id + 3], 0x0101010101010101)
            };

            total_simd = _mm256_add_epi8(total_simd, *((__m256i *)spread_words));
        }

        __mmask32 maj_bits = _mm256_cmpgt_epu8_mask(total_simd, threshold_simd);
        *((uint32_t*)&(dst_bytes[byte_id])) = maj_bits;
    }
}
#endif //__AVX512BW__

/// @brief AVX-256 implementation of threshold_into using a 1-Byte counter
void threshold_into_byte_avx256(word_t ** xs, uint8_t size, uint8_t threshold, word_t* dst) {
    __m256i threshold_simd = _mm256_set1_epi8(threshold);
    uint8_t** xs_bytes = (uint8_t**)xs;
    uint8_t* dst_bytes = (uint8_t*)dst;

    for (byte_iter_t byte_id = 0; byte_id < BYTES; byte_id += 4) {
        __m256i total_simd = _mm256_set1_epi8(0);

        for (u_int8_t i = 0; i < size; ++i) {
            uint8_t* bytes_i = xs_bytes[i];
            uint64_t spread_words[4] = {
                _pdep_u64(bytes_i[byte_id], 0x0101010101010101),
                _pdep_u64(bytes_i[byte_id + 1], 0x0101010101010101),
                _pdep_u64(bytes_i[byte_id + 2], 0x0101010101010101),
                _pdep_u64(bytes_i[byte_id + 3], 0x0101010101010101)
            };

            total_simd = _mm256_add_epi8(total_simd, *((__m256i *)spread_words));
        }

        uint64_t maj_words[4];
        *(__m256i *) maj_words = _mm256_cmpgt_epi8(total_simd, threshold_simd);

        dst_bytes[byte_id] = (uint8_t)_pext_u64(maj_words[0], 0x0101010101010101);
        dst_bytes[byte_id + 1] = (uint8_t)_pext_u64(maj_words[1], 0x0101010101010101);
        dst_bytes[byte_id + 2] = (uint8_t)_pext_u64(maj_words[2], 0x0101010101010101);
        dst_bytes[byte_id + 3] = (uint8_t)_pext_u64(maj_words[3], 0x0101010101010101);
    }
}

#if __AVX512BW__
    #define threshold_into_byte threshold_into_byte_avx512
#else
    #define threshold_into_byte threshold_into_byte_avx256
#endif //__AVX512BW__

/// @brief A generic implementation for threshold_into, that can use any size counter
template<typename N>
void threshold_into_generic(word_t ** xs, size_t size, N threshold, word_t *dst) {

    N totals[BITS];
    memset(totals, 0, BITS*sizeof(N));

    for (N i = 0; i < size; ++i) {
        word_t * x = xs[i];

        for (word_iter_t word_id = 0; word_id < WORDS; ++word_id) {
            bit_iter_t offset = word_id * BITS_PER_WORD;
            word_t word = x[word_id];
            for (bit_word_iter_t bit_id = 0; bit_id < BITS_PER_WORD; ++bit_id) {
                totals[offset + bit_id] += ((word >> bit_id) & 1);
            }
        }
    }

    for (word_iter_t word_id = 0; word_id < WORDS; ++word_id) {
        bit_iter_t offset = word_id * BITS_PER_WORD;
        word_t word = 0;
        for (bit_word_iter_t bit_id = 0; bit_id < BITS_PER_WORD; ++bit_id) {
            if (threshold < totals[offset + bit_id])
                word |= 1UL << bit_id;
        }
        dst[word_id] = word;
    }
}

#if __AVX512BW__
    #define threshold_into_short_wide threshold_into_short_avx512
#else
    #define threshold_into_short_wide threshold_into_generic<uint32_t>
#endif //__AVX512BW__

/// @brief Sets each result bit high if there are more than threshold 1 bits in the corresponding bit of the input vectors
/// @param xs array of `size` input vectors
/// @param size number of input vectors in xs
/// @param threshold threshold to count against
/// @param dst the hypervector to write the results into
void threshold_into(word_t ** xs, size_t size, size_t threshold, word_t* dst) {
    switch (size) {
        case 0 ... 255: threshold_into_byte(xs, size, size/2, dst); return;
        case 256 ... 650: threshold_into_short_avx256(xs, size, size/2, dst); return;
        //TODO: See note about changing the implementation of threshold_into_short_avx512
        //case 651 ... 10000: threshold_into_short_wide(xs, size, size/2, dst); return;
        default: threshold_into_generic<uint32_t>(xs, size, size/2, dst); return;
    }
}

/// @brief Sets each result bit high if there are more than threshold 1 bits in the corresponding bit of the input vectors
/// @param xs array of `size` input vectors
/// @param size number of input vectors in xs
/// @param threshold threshold to count against
/// @return returns a hypervector where each bit is set to 1 if the number of corresponding bits in xs exceeded threshold
word_t* threshold(word_t ** xs, size_t size, size_t threshold) {
    word_t* new_vec = bhv::empty();
    threshold_into(xs, size, threshold, new_vec);
    return new_vec;
}
