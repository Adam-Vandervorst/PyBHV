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

#if __AVX512BW__

/// @brief INTERNAL Counts an input cacheline worth of bits (64 Bytes = 512 bits) for up to 3 input hypervectors
/// @param xs array of input hypervectors
/// @param byte_offset offset (in bytes) into each hypervector.  Ideally this would be aligned to 64 Bytes
/// @param num_vectors the number of vectors in xs.  Maximum value of 3
/// @param out_counts Each counter is 2 bits, and there are 256 counters in each 512 bit AVX-512 vector, and there are 2 AVX-512 vectors.
///   Counters are interleaved between the output vectors.  For example, input bit positions indicateded by letters: H G F E D C B A,
///   lead to output bit positions: out_counts[0]: G  E  C  A, and out_counts[1]: H  F  D  B
inline void count_cacheline_for_3_input_hypervectors_avx512(word_t ** xs, size_t byte_offset, uint_fast8_t num_vectors, __m512i* out_counts) {

    const __m512i interleaved_bits = _mm512_set1_epi8(0x55);

    out_counts[0] = _mm512_set1_epi64(0);
    out_counts[1] = _mm512_set1_epi64(0);
    uint8_t** xs_bytes = (uint8_t**)xs;

    for (uint_fast8_t i = 0; i < num_vectors; i++) {
        uint8_t* input_vec_ptr = xs_bytes[i];
        __m512i input_bits = _mm512_stream_load_si512(input_vec_ptr + byte_offset);

        __m512i even_bits = _mm512_and_si512(input_bits, interleaved_bits);
        __m512i odd_bits = _mm512_and_si512(_mm512_srli_epi64(input_bits, 1), interleaved_bits);

        out_counts[0] = _mm512_add_epi8(out_counts[0], even_bits);
        out_counts[1] = _mm512_add_epi8(out_counts[1], odd_bits);
    }
}

/// @brief INTERNAL Counts an input cacheline worth of bits (64 Bytes = 512 bits) for up to 15 input hypervectors
/// @param xs array of input hypervectors
/// @param byte_offset offset (in bytes) into each hypervector.  Ideally this would be aligned to 64 Bytes
/// @param num_vectors the number of vectors in xs.  Maximum value of 15
/// @param out_counts Each counter is 4 bits, and there are 128 counters in each 512 bit AVX-512 vector, and there are 4 AVX-512 vectors
///  Counters are interleaved between output vectors, in a way that extends the interleaving described in
///   count_cacheline_for_3_input_hypervectors_avx512
inline void count_cacheline_for_15_input_hypervectors_avx512(word_t ** xs, size_t byte_offset, uint_fast8_t num_vectors, __m512i* out_counts) {
    const __m512i interleaved_pairs = _mm512_set1_epi8(0x33);

    out_counts[0] = _mm512_set1_epi64(0);
    out_counts[1] = _mm512_set1_epi64(0);
    out_counts[2] = _mm512_set1_epi64(0);
    out_counts[3] = _mm512_set1_epi64(0);

    for (uint_fast8_t i = 0; i < num_vectors; i+=3) {
        __m512i inner_counts[2];
        uint_fast16_t num_inputs = (i<num_vectors-2)? 3: num_vectors-i;

        count_cacheline_for_3_input_hypervectors_avx512(xs + i, byte_offset, num_inputs, inner_counts);

        //Expand the 2-bit counters into 4-bits, and add them to the running counters
        __m512i increment0 = _mm512_and_si512(inner_counts[0], interleaved_pairs);
        __m512i increment1 = _mm512_and_si512(inner_counts[1], interleaved_pairs);
        __m512i increment2 = _mm512_and_si512(_mm512_srli_epi64(inner_counts[0], 2), interleaved_pairs);
        __m512i increment3 = _mm512_and_si512(_mm512_srli_epi64(inner_counts[1], 2), interleaved_pairs);

        out_counts[0] = _mm512_add_epi8(out_counts[0], increment0);
        out_counts[1] = _mm512_add_epi8(out_counts[1], increment1);
        out_counts[2] = _mm512_add_epi8(out_counts[2], increment2);
        out_counts[3] = _mm512_add_epi8(out_counts[3], increment3);
    }
}

/// @brief INTERNAL Counts an input cacheline worth of bits (64 Bytes = 512 bits) for up to 255 input hypervectors
/// @param xs array of input hypervectors
/// @param byte_offset offset (in bytes) into each hypervector.  Ideally this would be aligned to 64 Bytes
/// @param num_vectors the number of vectors in xs.  Maximum value of 255
/// @param out_counts Each counter is 8 bits, and there are 64 counters in each 512 bit AVX-512 vector, and there
///  are 8 AVX-512 vectors.  Output counters are interleaved in a way that extends the interleaving described in
///  count_cacheline_for_3_input_hypervectors_avx512.  Counters can be un-scrambled with unscramble_byte_counters_avx512
inline void count_cacheline_for_255_input_hypervectors_avx512(word_t ** xs, size_t byte_offset, uint_fast8_t num_vectors, __m512i* out_counts) {
    const __m512i nibble_mask = _mm512_set1_epi8(0xF);

    //Zero the counters
    for (int i=0; i<8; i++) {
        out_counts[i] = _mm512_set1_epi64(0);
    }

    for (uint_fast8_t i = 0; i < num_vectors; i+=15) {
        __m512i inner_counts[4];
        uint_fast16_t num_inputs = (i<num_vectors-14)? 15: num_vectors-i;

        count_cacheline_for_15_input_hypervectors_avx512(xs + i, byte_offset, num_inputs, inner_counts);

        //Expand the 4-bit counters into 8-bits, and add them to the running counters
        __m512i increment0 = _mm512_and_si512(inner_counts[0], nibble_mask);
        __m512i increment1 = _mm512_and_si512(inner_counts[1], nibble_mask);
        __m512i increment2 = _mm512_and_si512(inner_counts[2], nibble_mask);
        __m512i increment3 = _mm512_and_si512(inner_counts[3], nibble_mask);
        __m512i increment4 = _mm512_and_si512(_mm512_srli_epi64(inner_counts[0], 4), nibble_mask);
        __m512i increment5 = _mm512_and_si512(_mm512_srli_epi64(inner_counts[1], 4), nibble_mask);
        __m512i increment6 = _mm512_and_si512(_mm512_srli_epi64(inner_counts[2], 4), nibble_mask);
        __m512i increment7 = _mm512_and_si512(_mm512_srli_epi64(inner_counts[3], 4), nibble_mask);

        out_counts[0] = _mm512_add_epi8(out_counts[0], increment0);
        out_counts[1] = _mm512_add_epi8(out_counts[1], increment1);
        out_counts[2] = _mm512_add_epi8(out_counts[2], increment2);
        out_counts[3] = _mm512_add_epi8(out_counts[3], increment3);
        out_counts[4] = _mm512_add_epi8(out_counts[4], increment4);
        out_counts[5] = _mm512_add_epi8(out_counts[5], increment5);
        out_counts[6] = _mm512_add_epi8(out_counts[6], increment6);
        out_counts[7] = _mm512_add_epi8(out_counts[7], increment7);
    }
}

/// @brief INTERNAL Unscrambles the counters returned from count_cacheline_for_255_input_hypervectors_avx512
/// @param scrambled_counts Each counter is 8 bits, and there are 64 counters in each 512 bit AVX-512 vector, and there are 8 AVX-512 vectors
/// @param out_counts Each counter is 8 bits, and there are 64 counters in each 512 bit AVX-512 vector, and there are 8 AVX-512 vectors
inline void unscramble_byte_counters_avx512(__m512i* scrambled_counts, __m512i* out_counts) {

    //Untangle the bytes, so the counters end up in the same order as the input bits
    __m512i unshuffle_l1[8];
    unshuffle_l1[0] = _mm512_unpacklo_epi8(scrambled_counts[0], scrambled_counts[1]);
    unshuffle_l1[1] = _mm512_unpackhi_epi8(scrambled_counts[0], scrambled_counts[1]);
    unshuffle_l1[2] = _mm512_unpacklo_epi8(scrambled_counts[2], scrambled_counts[3]);
    unshuffle_l1[3] = _mm512_unpackhi_epi8(scrambled_counts[2], scrambled_counts[3]);
    unshuffle_l1[4] = _mm512_unpacklo_epi8(scrambled_counts[4], scrambled_counts[5]);
    unshuffle_l1[5] = _mm512_unpackhi_epi8(scrambled_counts[4], scrambled_counts[5]);
    unshuffle_l1[6] = _mm512_unpacklo_epi8(scrambled_counts[6], scrambled_counts[7]);
    unshuffle_l1[7] = _mm512_unpackhi_epi8(scrambled_counts[6], scrambled_counts[7]);

    __m512i unshuffle_l2[8];
    unshuffle_l2[0] = _mm512_unpacklo_epi16(unshuffle_l1[0], unshuffle_l1[2]);
    unshuffle_l2[1] = _mm512_unpackhi_epi16(unshuffle_l1[0], unshuffle_l1[2]);
    unshuffle_l2[2] = _mm512_unpacklo_epi16(unshuffle_l1[4], unshuffle_l1[6]);
    unshuffle_l2[3] = _mm512_unpackhi_epi16(unshuffle_l1[4], unshuffle_l1[6]);
    unshuffle_l2[4] = _mm512_unpacklo_epi16(unshuffle_l1[1], unshuffle_l1[3]);
    unshuffle_l2[5] = _mm512_unpackhi_epi16(unshuffle_l1[1], unshuffle_l1[3]);
    unshuffle_l2[6] = _mm512_unpacklo_epi16(unshuffle_l1[5], unshuffle_l1[7]);
    unshuffle_l2[7] = _mm512_unpackhi_epi16(unshuffle_l1[5], unshuffle_l1[7]);

    __m512i unshuffle_l3[8];
    unshuffle_l3[0] = _mm512_unpacklo_epi32(unshuffle_l2[0], unshuffle_l2[2]);
    unshuffle_l3[1] = _mm512_unpackhi_epi32(unshuffle_l2[0], unshuffle_l2[2]);
    unshuffle_l3[2] = _mm512_unpacklo_epi32(unshuffle_l2[1], unshuffle_l2[3]);
    unshuffle_l3[3] = _mm512_unpackhi_epi32(unshuffle_l2[1], unshuffle_l2[3]);
    unshuffle_l3[4] = _mm512_unpacklo_epi32(unshuffle_l2[4], unshuffle_l2[6]);
    unshuffle_l3[5] = _mm512_unpackhi_epi32(unshuffle_l2[4], unshuffle_l2[6]);
    unshuffle_l3[6] = _mm512_unpacklo_epi32(unshuffle_l2[5], unshuffle_l2[7]);
    unshuffle_l3[7] = _mm512_unpackhi_epi32(unshuffle_l2[5], unshuffle_l2[7]);

    ((__m128i*)out_counts)[0] = ((__m128i*)unshuffle_l3)[0];
    ((__m128i*)out_counts)[1] = ((__m128i*)unshuffle_l3)[4];
    ((__m128i*)out_counts)[2] = ((__m128i*)unshuffle_l3)[8];
    ((__m128i*)out_counts)[3] = ((__m128i*)unshuffle_l3)[12];
    ((__m128i*)out_counts)[4] = ((__m128i*)unshuffle_l3)[16];
    ((__m128i*)out_counts)[5] = ((__m128i*)unshuffle_l3)[20];
    ((__m128i*)out_counts)[6] = ((__m128i*)unshuffle_l3)[24];
    ((__m128i*)out_counts)[7] = ((__m128i*)unshuffle_l3)[28];

    ((__m128i*)out_counts)[8] = ((__m128i*)unshuffle_l3)[1];
    ((__m128i*)out_counts)[9] = ((__m128i*)unshuffle_l3)[5];
    ((__m128i*)out_counts)[10] = ((__m128i*)unshuffle_l3)[9];
    ((__m128i*)out_counts)[11] = ((__m128i*)unshuffle_l3)[13];
    ((__m128i*)out_counts)[12] = ((__m128i*)unshuffle_l3)[17];
    ((__m128i*)out_counts)[13] = ((__m128i*)unshuffle_l3)[21];
    ((__m128i*)out_counts)[14] = ((__m128i*)unshuffle_l3)[25];
    ((__m128i*)out_counts)[15] = ((__m128i*)unshuffle_l3)[29];

    ((__m128i*)out_counts)[16] = ((__m128i*)unshuffle_l3)[2];
    ((__m128i*)out_counts)[17] = ((__m128i*)unshuffle_l3)[6];
    ((__m128i*)out_counts)[18] = ((__m128i*)unshuffle_l3)[10];
    ((__m128i*)out_counts)[19] = ((__m128i*)unshuffle_l3)[14];
    ((__m128i*)out_counts)[20] = ((__m128i*)unshuffle_l3)[18];
    ((__m128i*)out_counts)[21] = ((__m128i*)unshuffle_l3)[22];
    ((__m128i*)out_counts)[22] = ((__m128i*)unshuffle_l3)[26];
    ((__m128i*)out_counts)[23] = ((__m128i*)unshuffle_l3)[30];

    ((__m128i*)out_counts)[24] = ((__m128i*)unshuffle_l3)[3];
    ((__m128i*)out_counts)[25] = ((__m128i*)unshuffle_l3)[7];
    ((__m128i*)out_counts)[26] = ((__m128i*)unshuffle_l3)[11];
    ((__m128i*)out_counts)[27] = ((__m128i*)unshuffle_l3)[15];
    ((__m128i*)out_counts)[28] = ((__m128i*)unshuffle_l3)[19];
    ((__m128i*)out_counts)[29] = ((__m128i*)unshuffle_l3)[23];
    ((__m128i*)out_counts)[30] = ((__m128i*)unshuffle_l3)[27];
    ((__m128i*)out_counts)[31] = ((__m128i*)unshuffle_l3)[31];
}

/// @brief AVX-512 implementation of threshold_into using a 2-Byte counter
void threshold_into_short_avx512(word_t ** xs, int_fast16_t size, uint16_t threshold, word_t* dst) {
    __m512i threshold_simd = _mm512_set1_epi16(threshold);
    uint8_t* dst_bytes = (uint8_t*)dst;
    uint16_t counters[512];

    for (size_t byte_offset = 0; byte_offset < BYTES; byte_offset += 64) {

        //Clear out 16-bit counter registers
        memset(counters, 0, 512 * sizeof(uint16_t));

        //Loop over all input vectors, 255 at a time
        int_fast16_t i;
        for (i = 0; i < size; i += 255) {

            //Call (inline) the function to load one cache line of input bits from each input hypervector
            __m512i scrambled_counts[8];
            uint_fast16_t num_inputs = (i<size-254)? 255: size-i;
            count_cacheline_for_255_input_hypervectors_avx512(xs + i, byte_offset, num_inputs, scrambled_counts);

            //Unscramble the counters
            __m512i out_counts[8];
            unscramble_byte_counters_avx512(scrambled_counts, out_counts);

            //Expand the 8-bit counters into 16-bits, and add them to the running counters
            for (int_fast8_t out_i = 0; out_i < 8; out_i++) {
                __m512i increment0 = _mm512_cvtepu8_epi16(((__m256i*)&out_counts[out_i])[0]);
                __m512i increment1 = _mm512_cvtepu8_epi16(((__m256i*)&out_counts[out_i])[1]);
                *(__m512i*)(&counters[out_i * 64 + 0]) = _mm512_add_epi16(*(__m512i*)(&counters[out_i * 64 + 0]), increment0);
                *(__m512i*)(&counters[out_i * 64 + 32]) = _mm512_add_epi16(*(__m512i*)(&counters[out_i * 64 + 32]), increment1);
            }
        }

        //Now do thresholding, and output that block of 512bits of the output
        for (int_fast8_t out_i = 0; out_i < 16; out_i ++) {
            __mmask32 maj_bits = _mm512_cmpgt_epu16_mask(*(__m512i*)(&counters[out_i * 32]), threshold_simd);
            *((uint32_t*)(dst_bytes + byte_offset + (out_i * 4))) = maj_bits;
        }
    }
}

/// @brief AVX-512 implementation of threshold_into using a 1-Byte counter
void threshold_into_byte_avx512(word_t ** xs, uint8_t size, uint8_t threshold, word_t* dst) {
    __m512i threshold_simd = _mm512_set1_epi8(threshold);
    uint8_t* dst_bytes = (uint8_t*)dst;

    for (size_t byte_offset = 0; byte_offset < BYTES; byte_offset += 64) {

        //Call (inline) the function to load one cache line of input bits from each input hypervector
        __m512i scrambled_counts[8];
        count_cacheline_for_255_input_hypervectors_avx512(xs, byte_offset, size, scrambled_counts);

        //Unscramble the counters
        //FUTURE OPTIMIZATION: Performance could be improved on average 15-20% by performing the
        //threshold test first, and then unscrambling individual bits, rather than whole bytes.
        __m512i out_counts[8];
        unscramble_byte_counters_avx512(scrambled_counts, out_counts);

        //Do the threshold test, and compose out output bits
        __m512i out_bits;
        for (int i=0; i<8; i++) {
            __mmask64 maj_bits = _mm512_cmpgt_epu8_mask(out_counts[i], threshold_simd);
            ((uint64_t*)&out_bits)[i] = maj_bits;
        }

        //Store the results
        *((__m512i*)&(dst_bytes[byte_offset])) = out_bits;
    }
}

#endif //__AVX512BW__

/// @brief AVX-256 implementation of threshold_into using a 2-Byte counter
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
