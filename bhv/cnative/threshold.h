// This LUT is the exact same operation as `V = _pdep_u64(B, 0x0001000100010001);`
#define pdep_lut(V, B) switch(B) {\
    case 0: V=0x0; break;\
    case 1: V=0x1; break;\
    case 2: V=0x10000; break;\
    case 3: V=0x10001; break;\
    case 4: V=0x100000000; break;\
    case 5: V=0x100000001; break;\
    case 6: V=0x100010000; break;\
    case 7: V=0x100010001; break;\
    case 8: V=0x1000000000000; break;\
    case 9: V=0x1000000000001; break;\
    case 10: V=0x1000000010000; break;\
    case 11: V=0x1000000010001; break;\
    case 12: V=0x1000100000000; break;\
    case 13: V=0x1000100000001; break;\
    case 14: V=0x1000100010000; break;\
    case 15: V=0x1000100010001; break;\
    default: abort();\
}\


/// @brief INTERNAL Counts an input cacheline worth of bits (64 Bytes = 512 bits) for up to 15 input hypervectors
/// @param xs array of input hypervectors
/// @param byte_offset offset (in bytes) into each hypervector.  Ideally this would be aligned to 64 Bytes
/// @param num_vectors the number of vectors in xs.  Maximum value of 16
/// @param out_counts Each counter is 4 bits, and there are 128 counters in each 512 bit AVX-512 vector, and there are 4 AVX-512 vectors
inline void count_cacheline_for_15_input_hypervectors_avx512(word_t ** xs, size_t byte_offset, uint_fast8_t num_vectors, __m512i* out_counts) {
    out_counts[0] = _mm512_set1_epi64(0);
    out_counts[1] = _mm512_set1_epi64(0);
    out_counts[2] = _mm512_set1_epi64(0);
    out_counts[3] = _mm512_set1_epi64(0);
    uint8_t** xs_bytes = (uint8_t**)xs;

    for (uint_fast8_t i = 0; i < num_vectors; i++) {
        uint8_t* input_vec_ptr = xs_bytes[i];
        __m512i input_bits = _mm512_stream_load_si512(input_vec_ptr + byte_offset);

        for (uint_fast8_t out_i = 0; out_i < 4; out_i++) {
            uint_fast8_t input_offset = out_i * 8;

            __m512i increments;
            ((uint64_t*)&increments)[0] = _pdep_u64(((uint16_t*)&input_bits)[input_offset + 0], 0x1111111111111111);
            ((uint64_t*)&increments)[1] = _pdep_u64(((uint16_t*)&input_bits)[input_offset + 1], 0x1111111111111111);
            ((uint64_t*)&increments)[2] = _pdep_u64(((uint16_t*)&input_bits)[input_offset + 2], 0x1111111111111111);
            ((uint64_t*)&increments)[3] = _pdep_u64(((uint16_t*)&input_bits)[input_offset + 3], 0x1111111111111111);
            ((uint64_t*)&increments)[4] = _pdep_u64(((uint16_t*)&input_bits)[input_offset + 4], 0x1111111111111111);
            ((uint64_t*)&increments)[5] = _pdep_u64(((uint16_t*)&input_bits)[input_offset + 5], 0x1111111111111111);
            ((uint64_t*)&increments)[6] = _pdep_u64(((uint16_t*)&input_bits)[input_offset + 6], 0x1111111111111111);
            ((uint64_t*)&increments)[7] = _pdep_u64(((uint16_t*)&input_bits)[input_offset + 7], 0x1111111111111111);
            out_counts[out_i] = _mm512_add_epi8(out_counts[out_i], increments);
        }
    }
}

#if __AVX512BW__
/// @brief AVX-512 implementation of threshold_into using a 2-Byte counter
void threshold_into_short_avx512(word_t ** xs, int_fast16_t size, uint16_t threshold, word_t* dst) {
    __m512i threshold_simd = _mm512_set1_epi16(threshold);
    uint8_t* dst_bytes = (uint8_t*)dst;
    __m512i out_counts[4];
    uint16_t counters[512];

    for (size_t byte_offset = 0; byte_offset < BYTES; byte_offset += 64) {

        //Clear out 16-bit counter registers
        memset(counters, 0, 512 * sizeof(uint16_t));

        //Loop over all input vectors, 16 at a time
        int_fast16_t i;
        for (i = 0; i < size; i += 15) {

            //Call (inline) the function to load one cache line of input bits from each input hypervector
            uint_fast16_t num_inputs = (i<size-14)? 15: size-i;
            count_cacheline_for_15_input_hypervectors_avx512(xs + i, byte_offset, num_inputs, out_counts);

            //Expand the 4-bit counters into 16-bits, and add them to the running counters
            for (int_fast8_t out_i = 0; out_i < 4; out_i++) {

                __m512i increment0;
                ((uint64_t*)&increment0)[0] = _pdep_u64(((uint16_t*)&out_counts[out_i])[0], 0x000F000F000F000F);
                ((uint64_t*)&increment0)[1] = _pdep_u64(((uint16_t*)&out_counts[out_i])[1], 0x000F000F000F000F);
                ((uint64_t*)&increment0)[2] = _pdep_u64(((uint16_t*)&out_counts[out_i])[2], 0x000F000F000F000F);
                ((uint64_t*)&increment0)[3] = _pdep_u64(((uint16_t*)&out_counts[out_i])[3], 0x000F000F000F000F);
                ((uint64_t*)&increment0)[4] = _pdep_u64(((uint16_t*)&out_counts[out_i])[4], 0x000F000F000F000F);
                ((uint64_t*)&increment0)[5] = _pdep_u64(((uint16_t*)&out_counts[out_i])[5], 0x000F000F000F000F);
                ((uint64_t*)&increment0)[6] = _pdep_u64(((uint16_t*)&out_counts[out_i])[6], 0x000F000F000F000F);
                ((uint64_t*)&increment0)[7] = _pdep_u64(((uint16_t*)&out_counts[out_i])[7], 0x000F000F000F000F);
                *(__m512i*)(&counters[out_i * 128 + 0]) = _mm512_add_epi16(*(__m512i*)(&counters[out_i * 128 + 0]), increment0);

                __m512i increment1;
                ((uint64_t*)&increment1)[0] = _pdep_u64(((uint16_t*)&out_counts[out_i])[8], 0x000F000F000F000F);
                ((uint64_t*)&increment1)[1] = _pdep_u64(((uint16_t*)&out_counts[out_i])[9], 0x000F000F000F000F);
                ((uint64_t*)&increment1)[2] = _pdep_u64(((uint16_t*)&out_counts[out_i])[10], 0x000F000F000F000F);
                ((uint64_t*)&increment1)[3] = _pdep_u64(((uint16_t*)&out_counts[out_i])[11], 0x000F000F000F000F);
                ((uint64_t*)&increment1)[4] = _pdep_u64(((uint16_t*)&out_counts[out_i])[12], 0x000F000F000F000F);
                ((uint64_t*)&increment1)[5] = _pdep_u64(((uint16_t*)&out_counts[out_i])[13], 0x000F000F000F000F);
                ((uint64_t*)&increment1)[6] = _pdep_u64(((uint16_t*)&out_counts[out_i])[14], 0x000F000F000F000F);
                ((uint64_t*)&increment1)[7] = _pdep_u64(((uint16_t*)&out_counts[out_i])[15], 0x000F000F000F000F);
                *(__m512i*)(&counters[out_i * 128 + 32]) = _mm512_add_epi16(*(__m512i*)(&counters[out_i * 128 + 32]), increment1);

                __m512i increment2;
                ((uint64_t*)&increment2)[0] = _pdep_u64(((uint16_t*)&out_counts[out_i])[16], 0x000F000F000F000F);
                ((uint64_t*)&increment2)[1] = _pdep_u64(((uint16_t*)&out_counts[out_i])[17], 0x000F000F000F000F);
                ((uint64_t*)&increment2)[2] = _pdep_u64(((uint16_t*)&out_counts[out_i])[18], 0x000F000F000F000F);
                ((uint64_t*)&increment2)[3] = _pdep_u64(((uint16_t*)&out_counts[out_i])[19], 0x000F000F000F000F);
                ((uint64_t*)&increment2)[4] = _pdep_u64(((uint16_t*)&out_counts[out_i])[20], 0x000F000F000F000F);
                ((uint64_t*)&increment2)[5] = _pdep_u64(((uint16_t*)&out_counts[out_i])[21], 0x000F000F000F000F);
                ((uint64_t*)&increment2)[6] = _pdep_u64(((uint16_t*)&out_counts[out_i])[22], 0x000F000F000F000F);
                ((uint64_t*)&increment2)[7] = _pdep_u64(((uint16_t*)&out_counts[out_i])[23], 0x000F000F000F000F);
                *(__m512i*)(&counters[out_i * 128 + 64]) = _mm512_add_epi16(*(__m512i*)(&counters[out_i * 128 + 64]), increment2);

                __m512i increment3;
                ((uint64_t*)&increment3)[0] = _pdep_u64(((uint16_t*)&out_counts[out_i])[24], 0x000F000F000F000F);
                ((uint64_t*)&increment3)[1] = _pdep_u64(((uint16_t*)&out_counts[out_i])[25], 0x000F000F000F000F);
                ((uint64_t*)&increment3)[2] = _pdep_u64(((uint16_t*)&out_counts[out_i])[26], 0x000F000F000F000F);
                ((uint64_t*)&increment3)[3] = _pdep_u64(((uint16_t*)&out_counts[out_i])[27], 0x000F000F000F000F);
                ((uint64_t*)&increment3)[4] = _pdep_u64(((uint16_t*)&out_counts[out_i])[28], 0x000F000F000F000F);
                ((uint64_t*)&increment3)[5] = _pdep_u64(((uint16_t*)&out_counts[out_i])[29], 0x000F000F000F000F);
                ((uint64_t*)&increment3)[6] = _pdep_u64(((uint16_t*)&out_counts[out_i])[30], 0x000F000F000F000F);
                ((uint64_t*)&increment3)[7] = _pdep_u64(((uint16_t*)&out_counts[out_i])[31], 0x000F000F000F000F);
                *(__m512i*)(&counters[out_i * 128 + 96]) = _mm512_add_epi16(*(__m512i*)(&counters[out_i * 128 + 96]), increment3);
            }
        }

        //Now do thresholding, and output that block of 512bits of the output
        for (int_fast8_t out_i = 0; out_i < 16; out_i ++) {
            __mmask32 maj_bits = _mm512_cmpgt_epu16_mask(*(__m512i*)(&counters[out_i * 32]), threshold_simd);
            *((uint32_t*)(dst_bytes + byte_offset + (out_i * 4))) = maj_bits;
        }
    }
}
#endif //__AVX512BW__

/// @brief AVX-256 implementation of threshold_into using a 2-Byte counter
/// @note The AVX-256 implementation is faster than threshold_into_short_avx512 for smaller
/// values or N, even when AVX-512  is available, probably on account of less cache pressure.
void threshold_into_short_avx2(word_t ** xs, uint16_t size, uint16_t threshold, word_t* dst) {
    __m256i threshold_simd = _mm256_set1_epi16(threshold);
    uint8_t** xs_bytes = (uint8_t**)xs;
    uint8_t* dst_bytes = (uint8_t*)dst;

    for (byte_iter_t byte_id = 0; byte_id < BYTES; byte_id += 2) {
        __m256i total_simd = _mm256_set1_epi16(0);

        for (uint16_t i = 0; i < size; ++i) {
            uint8_t* bytes_i = xs_bytes[i];

            //NOTE: Experimentally, a mix of pdep instructions and LUT lookups end up being
            // faster than using either exclusively, probably on account of limited throughput
            // through some hardware unit
            uint64_t spread_words[4];
            //pdep_lut(spread_words[0], bytes_i[byte_id] & 0xF);
            spread_words[0] = _pdep_u64(bytes_i[byte_id], 0x0001000100010001);
            pdep_lut(spread_words[1], bytes_i[byte_id] >> 4);
            //pdep_lut(spread_words[2], bytes_i[byte_id + 1] & 0xF);
            spread_words[2] = _pdep_u64(bytes_i[byte_id+1], 0x0001000100010001);
            pdep_lut(spread_words[3], bytes_i[byte_id + 1] >> 4);

            total_simd = _mm256_add_epi16(total_simd, *((__m256i *)spread_words));
        }

        uint64_t maj_words[4];
        *(__m256i *) maj_words = _mm256_cmpgt_epi16(total_simd, threshold_simd);

        dst_bytes[byte_id] = (uint8_t)_pext_u64(maj_words[0], 0x0001000100010001) | (uint8_t)_pext_u64(maj_words[1], 0x0001000100010001) << 4;
        dst_bytes[byte_id + 1] = (uint8_t)_pext_u64(maj_words[2], 0x0001000100010001) | (uint8_t)_pext_u64(maj_words[3], 0x0001000100010001) << 4;
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
void threshold_into_byte_avx2(word_t ** xs, uint8_t size, uint8_t threshold, word_t* dst) {
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
    #define threshold_into_byte threshold_into_byte_avx2
#endif //__AVX512BW__

/// @brief A generic implementation for threshold_into, that can use any size counter
template<typename N>
void threshold_into_reference(word_t ** xs, size_t size, N threshold, word_t *dst) {

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
    #define threshold_into_short threshold_into_short_avx512
#else
    #define threshold_into_short threshold_into_reference<uint32_t>
#endif //__AVX512BW__

/// @brief Sets each result bit high if there are more than threshold 1 bits in the corresponding bit of the input vectors
/// @param xs array of `size` input vectors
/// @param size number of input vectors in xs
/// @param threshold threshold to count against
/// @param dst the hypervector to write the results into
void threshold_into(word_t ** xs, size_t size, size_t threshold, word_t* dst) {
    //TODO: Should we have a path for small sizes?
    if (size < 256) { threshold_into_byte(xs, size, threshold, dst); return; }
    if (size < 65535) { threshold_into_short(xs, size, threshold, dst); return; }
    threshold_into_reference<uint32_t>(xs, size, threshold, dst); return;
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
