
/// @brief A generic implementation for (exclusive) threshold_into, that can use any size counter
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
/// @brief INTERNAL Counts an input cacheline worth of bits (64 Bytes = 512 bits) for 1 input hypervector
/// @param xs pointer to pointer to the input hypervector data
/// @param byte_offset offset (in bytes) into each hypervector.  This must be aligned to 64 Bytes
/// @param out_counts Each counter is 2 bits, and there are 256 counters in each 512 bit AVX-512 vector, and there are 2 AVX-512 vectors.
///   Counters are interleaved between the output vectors.  For example, input bit positions indicateded by letters: H G F E D C B A,
///   lead to output bit positions: out_counts[0]: G  E  C  A, and out_counts[1]: H  F  D  B
inline void add_counts_from_cacheline_for_1_input_hypervector_avx512(word_t ** xs, size_t byte_offset, __m512i* out_counts) {
    const __m512i interleaved_bits = _mm512_set1_epi8(0x55);
    uint8_t* xs_bytes = *((uint8_t**)xs);
    __m512i input_bits = _mm512_loadu_si512(xs_bytes + byte_offset);

    __m512i even_bits = _mm512_and_si512(input_bits, interleaved_bits);
    __m512i odd_bits = _mm512_and_si512(_mm512_srli_epi64(input_bits, 1), interleaved_bits);

    out_counts[0] = _mm512_add_epi8(out_counts[0], even_bits);
    out_counts[1] = _mm512_add_epi8(out_counts[1], odd_bits);
}

/// @brief INTERNAL Counts an input cacheline worth of bits (64 Bytes = 512 bits) for up to 15 input hypervectors
/// @param xs array of input hypervectors
/// @param byte_offset offset (in bytes) into each hypervector.  Ideally this would be aligned to 64 Bytes
/// @param num_vectors the number of vectors in xs.  Maximum value of 15
/// @param out_counts Each counter is 4 bits, and there are 128 counters in each 512 bit AVX-512 vector, and there are 4 AVX-512 vectors
///  Counters are interleaved between output vectors, in a way that extends the interleaving described in
///   add_counts_from_cacheline_for_1_input_hypervector_avx512
inline void count_cacheline_for_15_input_hypervectors_avx512(word_t ** xs, size_t byte_offset, uint_fast8_t num_vectors, __m512i* out_counts) {
    const __m512i interleaved_pairs = _mm512_set1_epi8(0x33);

    out_counts[0] = _mm512_set1_epi64(0);
    out_counts[1] = _mm512_set1_epi64(0);
    out_counts[2] = _mm512_set1_epi64(0);
    out_counts[3] = _mm512_set1_epi64(0);

    __m512i inner_counts[2];
    for (uint_fast8_t i = 0; i < num_vectors-2; i+=3) {
        inner_counts[0] = _mm512_set1_epi64(0);
        inner_counts[1] = _mm512_set1_epi64(0);

        add_counts_from_cacheline_for_1_input_hypervector_avx512(xs + i, byte_offset, inner_counts);
        add_counts_from_cacheline_for_1_input_hypervector_avx512(xs + i + 1, byte_offset, inner_counts);
        add_counts_from_cacheline_for_1_input_hypervector_avx512(xs + i + 2, byte_offset, inner_counts);

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
    if (num_vectors % 3 == 0) return;

    // Mop up the straggler bits
    inner_counts[0] = _mm512_set1_epi64(0);
    inner_counts[1] = _mm512_set1_epi64(0);
    for (uint_fast8_t i = (num_vectors/3)*3; i < num_vectors; i++) {
        add_counts_from_cacheline_for_1_input_hypervector_avx512(xs + i, byte_offset, inner_counts);
    }
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

/// @brief INTERNAL Counts an input cacheline worth of bits (64 Bytes = 512 bits) for up to 255 input hypervectors
/// @param xs array of input hypervectors
/// @param byte_offset offset (in bytes) into each hypervector.  Ideally this would be aligned to 64 Bytes
/// @param num_vectors the number of vectors in xs.  Maximum value of 255
/// @param out_counts Each counter is 8 bits, and there are 64 counters in each 512 bit AVX-512 vector, and there
///  are 8 AVX-512 vectors.  Output counters are interleaved in a way that extends the interleaving described in
///  add_counts_from_cacheline_for_1_input_hypervector_avx512.  Counters can be un-scrambled with unscramble_byte_counters_avx512
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
    uint16_t counters[BITS];

    //Clear out 16-bit counters
    memset(counters, 0, BITS * sizeof(uint16_t));

    //Loop over all input vectors, 255 at a time
    for (int_fast16_t i = 0; i < size; i += 255) {
        uint_fast16_t num_inputs = (i<size-254)? 255: size-i;

        size_t cur_counters = 0;
        for (size_t byte_offset = 0; byte_offset < BYTES; byte_offset += 64) {

            //Call (inline) the function to load one cache line of input bits from each input hypervector
            __m512i scrambled_counts[8];
            count_cacheline_for_255_input_hypervectors_avx512(xs + i, byte_offset, num_inputs, scrambled_counts);

            //Unscramble the counters
            __m512i out_counts[8];
            unscramble_byte_counters_avx512(scrambled_counts, out_counts);

            //Expand the 8-bit counters into 16-bits, and add them to the running counters
            for (int_fast8_t out_i = 0; out_i < 8; out_i++) {
                __m512i increment0 = _mm512_cvtepu8_epi16(((__m256i*)&out_counts[out_i])[0]);
                __m512i increment1 = _mm512_cvtepu8_epi16(((__m256i*)&out_counts[out_i])[1]);
                _mm512_storeu_si512((__m512i*)(&counters[cur_counters + 0]), _mm512_add_epi16(*(__m512i*)(&counters[cur_counters + 0]), increment0));
                _mm512_storeu_si512((__m512i*)(&counters[cur_counters + 32]), _mm512_add_epi16(*(__m512i*)(&counters[cur_counters + 32]), increment1));
                cur_counters += 64;
            }
        }
    }

    //Now do thresholding and output
    for (size_t i = 0; i < BITS/32; i++) {
        __mmask32 maj_bits = _mm512_cmpgt_epu16_mask(_mm512_loadu_si512(counters + i * 32), threshold_simd);
        *((uint32_t*)(dst_bytes + (i * 4))) = maj_bits;
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
        _mm512_storeu_si512((__m512i*)(dst_bytes + byte_offset), out_bits);
    }
}

/// @brief AVX-512 implementation of threshold_into using a 4-Byte counter
void threshold_into_32bit_avx512(word_t ** xs, uint32_t size, uint32_t threshold, word_t* dst) {
    __m512i threshold_simd = _mm512_set1_epi32(threshold);
    uint8_t* dst_bytes = (uint8_t*)dst;
    uint32_t counters[BITS];

    //Clear out the counters
    memset(counters, 0, BITS * sizeof(uint32_t));

    //Loop over all input vectors, 255 at a time
    for (uint_fast32_t i = 0; i < size; i += 255) {
        uint_fast32_t num_inputs = (i<size-254)? 255: size-i;

        size_t cur_counters = 0;
        for (size_t byte_offset = 0; byte_offset < BYTES; byte_offset += 64) {

            //Call (inline) the function to load one cache line of input bits from each input hypervector
            __m512i scrambled_counts[8];
            count_cacheline_for_255_input_hypervectors_avx512(xs + i, byte_offset, num_inputs, scrambled_counts);

            //Unscramble the counters
            __m512i out_counts[8];
            unscramble_byte_counters_avx512(scrambled_counts, out_counts);

            //Expand the 8-bit counters into 32-bits, and add them to the running counters
            for (int_fast8_t out_i = 0; out_i < 8; out_i++) {
                __m512i increment0 = _mm512_cvtepu8_epi32(((__m128i*)&out_counts[out_i])[0]);
                __m512i increment1 = _mm512_cvtepu8_epi32(((__m128i*)&out_counts[out_i])[1]);
                __m512i increment2 = _mm512_cvtepu8_epi32(((__m128i*)&out_counts[out_i])[2]);
                __m512i increment3 = _mm512_cvtepu8_epi32(((__m128i*)&out_counts[out_i])[3]);
                _mm512_storeu_si512((__m512i*)(&counters[cur_counters + 0]), _mm512_add_epi32(*(__m512i*)(&counters[cur_counters + 0]), increment0));
                _mm512_storeu_si512((__m512i*)(&counters[cur_counters + 16]), _mm512_add_epi32(*(__m512i*)(&counters[cur_counters + 16]), increment1));
                _mm512_storeu_si512((__m512i*)(&counters[cur_counters + 32]), _mm512_add_epi32(*(__m512i*)(&counters[cur_counters + 32]), increment2));
                _mm512_storeu_si512((__m512i*)(&counters[cur_counters + 48]), _mm512_add_epi32(*(__m512i*)(&counters[cur_counters + 48]), increment3));
                cur_counters += 64;
            }
        }
    }

    //Now do thresholding and output
    for (size_t i = 0; i < BITS/16; i++) {
        __mmask16 maj_bits = _mm512_cmpgt_epu32_mask(_mm512_loadu_si512(counters + i * 16), threshold_simd);
        *((uint16_t*)(dst_bytes + (i * 2))) = maj_bits;
    }
}

/// @brief Sets each result bit high if there are more than threshold 1 bits in the corresponding bit of the input vectors
/// @param xs array of `size` input vectors
/// @param size number of input vectors in xs
/// @param threshold threshold to count against
/// @param dst the hypervector to write the results into
void threshold_into_avx512(word_t ** xs, size_t size, size_t threshold, word_t* dst) {
    //FUTURE OPTIMIZATION: Should we have a path for smaller sizes?  Currently the main user of
    // threshold_into() is true_majority(), and it has dedicated code for cases where n <= 21
    if (size < 256) { threshold_into_byte_avx512(xs, size, threshold, dst); return; }
    if (size < 65536) { threshold_into_short_avx512(xs, size, threshold, dst); return; }
    threshold_into_32bit_avx512(xs, size, threshold, dst);
}
#endif //__AVX512BW__

#ifdef __AVX2__

template<uint8_t size, uint8_t t>
void logic_threshold_into_avx2(word_t **xs, word_t *target) {
    constexpr uint8_t t_ = size - t - 1;
    __m256i grid[t + 1][t_ + 1];

    for (word_iter_t word_id = 0; word_id < WORDS; word_id += 4) {
        word_t *x = xs[size - 1];
        grid[t][t_] = _mm256_loadu_si256((__m256i *) (x + word_id));

        for (uint8_t i = 0; i < t; ++i) {
            x = xs[size - i - 2];
            __m256i chunk = _mm256_loadu_si256((__m256i *) (x + word_id));

            grid[t - i - 1][t_] = grid[t - i][t_] & chunk;
        }

        for (uint8_t i = 0; i < t_; ++i) {
            x = xs[size - i - 2];
            __m256i chunk = _mm256_loadu_si256((__m256i *) (x + word_id));

            grid[t][t_ - i - 1] = grid[t][t_ - i] | chunk;
        }

        for (uint8_t i = t - 1; i < t; --i)
            for (uint8_t j = t_ - 1; j < t_; --j) {
                x = xs[i + j];
                __m256i chunk = _mm256_loadu_si256((__m256i *) (x + word_id));

                grid[i][j] = grid[i][j + 1] ^ (chunk & (grid[i][j + 1] ^ grid[i + 1][j])); // select
            }

        _mm256_storeu_si256((__m256i *) (target + word_id), grid[0][0]);
    }
}

/// @brief INTERNAL Counts 256 input bits (32 Bytes) for 1 input hypervector
/// @param xs pointer to pointer to input hypervector data
/// @param byte_offset offset (in bytes) into each hypervector.  Must be aligned to 32 Bytes
/// @param out_counts Each counter is 2 bits, and there are 128 counters in each 256 bit AVX2 vector, so there are 2 AVX2 vectors.
inline void add_counts_from_half_cacheline_for_1_input_hypervector_avx2(word_t ** xs, size_t byte_offset, __m256i* out_counts) {
    const __m256i interleaved_bits = _mm256_set1_epi8(0x55);
    uint8_t* xs_bytes = *((uint8_t**)xs);
    __m256i input_bits = _mm256_loadu_si256((__m256i*)(xs_bytes + byte_offset));

    __m256i even_bits = _mm256_and_si256(input_bits, interleaved_bits);
    __m256i odd_bits = _mm256_and_si256(_mm256_srli_epi64(input_bits, 1), interleaved_bits);

    out_counts[0] = _mm256_add_epi8(out_counts[0], even_bits);
    out_counts[1] = _mm256_add_epi8(out_counts[1], odd_bits);
}

/// @brief INTERNAL Counts 256 input bits (32 Bytes) for up to 15 input hypervectors
/// @param xs array of input hypervectors
/// @param byte_offset offset (in bytes) into each hypervector.  This must be aligned to 32 Bytes
/// @param num_vectors the number of vectors in xs.  Maximum value of 15
/// @param out_counts Each counter is 4 bits, and there are 64 counters in each 256 bit AVX2 vector, and there are 4 AVX2 vectors
inline void count_half_cacheline_for_15_input_hypervectors_avx2(word_t ** xs, size_t byte_offset, uint_fast8_t num_vectors, __m256i* out_counts) {
    const __m256i interleaved_pairs = _mm256_set1_epi8(0x33);

    out_counts[0] = _mm256_set1_epi64x(0);
    out_counts[1] = _mm256_set1_epi64x(0);
    out_counts[2] = _mm256_set1_epi64x(0);
    out_counts[3] = _mm256_set1_epi64x(0);

    __m256i inner_counts[2];
    for (uint_fast8_t i = 0; i < num_vectors-2; i+=3) {
        inner_counts[0] = _mm256_set1_epi64x(0);
        inner_counts[1] = _mm256_set1_epi64x(0);

        add_counts_from_half_cacheline_for_1_input_hypervector_avx2(xs + i, byte_offset, inner_counts);
        add_counts_from_half_cacheline_for_1_input_hypervector_avx2(xs + i + 1, byte_offset, inner_counts);
        add_counts_from_half_cacheline_for_1_input_hypervector_avx2(xs + i + 2, byte_offset, inner_counts);

        //Expand the 2-bit counters into 4-bits, and add them to the running counters
        __m256i increment0 = _mm256_and_si256(inner_counts[0], interleaved_pairs);
        __m256i increment1 = _mm256_and_si256(inner_counts[1], interleaved_pairs);
        __m256i increment2 = _mm256_and_si256(_mm256_srli_epi64(inner_counts[0], 2), interleaved_pairs);
        __m256i increment3 = _mm256_and_si256(_mm256_srli_epi64(inner_counts[1], 2), interleaved_pairs);
        out_counts[0] = _mm256_add_epi8(out_counts[0], increment0);
        out_counts[1] = _mm256_add_epi8(out_counts[1], increment1);
        out_counts[2] = _mm256_add_epi8(out_counts[2], increment2);
        out_counts[3] = _mm256_add_epi8(out_counts[3], increment3);
    }
    if (num_vectors % 3 == 0) return;

    // Mop up the straggler bits
    inner_counts[0] = _mm256_set1_epi64x(0);
    inner_counts[1] = _mm256_set1_epi64x(0);
    for (uint_fast8_t i = (num_vectors/3)*3; i < num_vectors; i++) {
        add_counts_from_half_cacheline_for_1_input_hypervector_avx2(xs + i, byte_offset, inner_counts);
    }
    //Expand the 2-bit counters into 4-bits, and add them to the running counters
    __m256i increment0 = _mm256_and_si256(inner_counts[0], interleaved_pairs);
    __m256i increment1 = _mm256_and_si256(inner_counts[1], interleaved_pairs);
    __m256i increment2 = _mm256_and_si256(_mm256_srli_epi64(inner_counts[0], 2), interleaved_pairs);
    __m256i increment3 = _mm256_and_si256(_mm256_srli_epi64(inner_counts[1], 2), interleaved_pairs);
    out_counts[0] = _mm256_add_epi8(out_counts[0], increment0);
    out_counts[1] = _mm256_add_epi8(out_counts[1], increment1);
    out_counts[2] = _mm256_add_epi8(out_counts[2], increment2);
    out_counts[3] = _mm256_add_epi8(out_counts[3], increment3);
}

/// @brief INTERNAL Counts 256 input bits (32 Bytes) for up to 255 input hypervectors
/// @param xs array of input hypervectors
/// @param byte_offset offset (in bytes) into each hypervector.  This must be aligned to 32 Bytes
/// @param num_vectors the number of vectors in xs.  Maximum value of 255
/// @param out_counts Each counter is 8 bits, and there are 32 counters in each 256 bit AVX2 vector, and there are 8 AVX2 vectors
inline void count_half_cacheline_for_255_input_hypervectors_avx2(word_t ** xs, size_t byte_offset, uint_fast8_t num_vectors, __m256i* out_counts) {
    const __m256i nibble_mask = _mm256_set1_epi8(0xF);

    //Zero the counters
    for (int i=0; i<8; i++) {
        out_counts[i] = _mm256_set1_epi64x(0);
    }

    for (uint_fast8_t i = 0; i < num_vectors; i+=15) {
        __m256i inner_counts[4];
        uint_fast16_t num_inputs = (i<num_vectors-14)? 15: num_vectors-i;

        count_half_cacheline_for_15_input_hypervectors_avx2(xs + i, byte_offset, num_inputs, inner_counts);

        //Expand the 4-bit counters into 8-bits, and add them to the running counters
        __m256i increment0 = _mm256_and_si256(inner_counts[0], nibble_mask);
        __m256i increment1 = _mm256_and_si256(inner_counts[1], nibble_mask);
        __m256i increment2 = _mm256_and_si256(inner_counts[2], nibble_mask);
        __m256i increment3 = _mm256_and_si256(inner_counts[3], nibble_mask);
        __m256i increment4 = _mm256_and_si256(_mm256_srli_epi64(inner_counts[0], 4), nibble_mask);
        __m256i increment5 = _mm256_and_si256(_mm256_srli_epi64(inner_counts[1], 4), nibble_mask);
        __m256i increment6 = _mm256_and_si256(_mm256_srli_epi64(inner_counts[2], 4), nibble_mask);
        __m256i increment7 = _mm256_and_si256(_mm256_srli_epi64(inner_counts[3], 4), nibble_mask);

        out_counts[0] = _mm256_add_epi8(out_counts[0], increment0);
        out_counts[1] = _mm256_add_epi8(out_counts[1], increment1);
        out_counts[2] = _mm256_add_epi8(out_counts[2], increment2);
        out_counts[3] = _mm256_add_epi8(out_counts[3], increment3);
        out_counts[4] = _mm256_add_epi8(out_counts[4], increment4);
        out_counts[5] = _mm256_add_epi8(out_counts[5], increment5);
        out_counts[6] = _mm256_add_epi8(out_counts[6], increment6);
        out_counts[7] = _mm256_add_epi8(out_counts[7], increment7);
    }
}

/// @brief INTERNAL Unscrambles the counters returned from count_half_cacheline_for_255_input_hypervectors_avx2
/// @param scrambled_counts Each counter is 8 bits, and there are 32 counters in each 256 bit AVX2 vector, and there are 8 AVX2 vectors
/// @param out_counts Each counter is 8 bits, and there are 32 counters in each 256 bit AVX2 vector, and there are 8 AVX2 vectors
inline void unscramble_byte_counters_avx2(__m256i* scrambled_counts, __m256i* out_counts) {

    //Untangle the bytes, so the counters end up in the same order as the input bits
    __m256i unshuffle_l1[8];
    unshuffle_l1[0] = _mm256_unpacklo_epi8(scrambled_counts[0], scrambled_counts[1]);
    unshuffle_l1[1] = _mm256_unpackhi_epi8(scrambled_counts[0], scrambled_counts[1]);
    unshuffle_l1[2] = _mm256_unpacklo_epi8(scrambled_counts[2], scrambled_counts[3]);
    unshuffle_l1[3] = _mm256_unpackhi_epi8(scrambled_counts[2], scrambled_counts[3]);
    unshuffle_l1[4] = _mm256_unpacklo_epi8(scrambled_counts[4], scrambled_counts[5]);
    unshuffle_l1[5] = _mm256_unpackhi_epi8(scrambled_counts[4], scrambled_counts[5]);
    unshuffle_l1[6] = _mm256_unpacklo_epi8(scrambled_counts[6], scrambled_counts[7]);
    unshuffle_l1[7] = _mm256_unpackhi_epi8(scrambled_counts[6], scrambled_counts[7]);

    __m256i unshuffle_l2[8];
    unshuffle_l2[0] = _mm256_unpacklo_epi16(unshuffle_l1[0], unshuffle_l1[2]);
    unshuffle_l2[1] = _mm256_unpackhi_epi16(unshuffle_l1[0], unshuffle_l1[2]);
    unshuffle_l2[2] = _mm256_unpacklo_epi16(unshuffle_l1[4], unshuffle_l1[6]);
    unshuffle_l2[3] = _mm256_unpackhi_epi16(unshuffle_l1[4], unshuffle_l1[6]);
    unshuffle_l2[4] = _mm256_unpacklo_epi16(unshuffle_l1[1], unshuffle_l1[3]);
    unshuffle_l2[5] = _mm256_unpackhi_epi16(unshuffle_l1[1], unshuffle_l1[3]);
    unshuffle_l2[6] = _mm256_unpacklo_epi16(unshuffle_l1[5], unshuffle_l1[7]);
    unshuffle_l2[7] = _mm256_unpackhi_epi16(unshuffle_l1[5], unshuffle_l1[7]);

    __m256i unshuffle_l3[8];
    unshuffle_l3[0] = _mm256_unpacklo_epi32(unshuffle_l2[0], unshuffle_l2[2]);
    unshuffle_l3[1] = _mm256_unpackhi_epi32(unshuffle_l2[0], unshuffle_l2[2]);
    unshuffle_l3[2] = _mm256_unpacklo_epi32(unshuffle_l2[1], unshuffle_l2[3]);
    unshuffle_l3[3] = _mm256_unpackhi_epi32(unshuffle_l2[1], unshuffle_l2[3]);
    unshuffle_l3[4] = _mm256_unpacklo_epi32(unshuffle_l2[4], unshuffle_l2[6]);
    unshuffle_l3[5] = _mm256_unpackhi_epi32(unshuffle_l2[4], unshuffle_l2[6]);
    unshuffle_l3[6] = _mm256_unpacklo_epi32(unshuffle_l2[5], unshuffle_l2[7]);
    unshuffle_l3[7] = _mm256_unpackhi_epi32(unshuffle_l2[5], unshuffle_l2[7]);

    ((__m128i*)out_counts)[0] = ((__m128i*)unshuffle_l3)[0];
    ((__m128i*)out_counts)[1] = ((__m128i*)unshuffle_l3)[2];
    ((__m128i*)out_counts)[2] = ((__m128i*)unshuffle_l3)[4];
    ((__m128i*)out_counts)[3] = ((__m128i*)unshuffle_l3)[6];
    ((__m128i*)out_counts)[4] = ((__m128i*)unshuffle_l3)[8];
    ((__m128i*)out_counts)[5] = ((__m128i*)unshuffle_l3)[10];
    ((__m128i*)out_counts)[6] = ((__m128i*)unshuffle_l3)[12];
    ((__m128i*)out_counts)[7] = ((__m128i*)unshuffle_l3)[14];

    ((__m128i*)out_counts)[8] = ((__m128i*)unshuffle_l3)[1];
    ((__m128i*)out_counts)[9] = ((__m128i*)unshuffle_l3)[3];
    ((__m128i*)out_counts)[10] = ((__m128i*)unshuffle_l3)[5];
    ((__m128i*)out_counts)[11] = ((__m128i*)unshuffle_l3)[7];
    ((__m128i*)out_counts)[12] = ((__m128i*)unshuffle_l3)[9];
    ((__m128i*)out_counts)[13] = ((__m128i*)unshuffle_l3)[11];
    ((__m128i*)out_counts)[14] = ((__m128i*)unshuffle_l3)[13];
    ((__m128i*)out_counts)[15] = ((__m128i*)unshuffle_l3)[15];
}

/// @brief AVX2 implementation of threshold_into using a 2-Byte counter
void threshold_into_short_avx2(word_t ** xs, int_fast16_t size, uint16_t threshold, word_t* dst) {
    const __m256i signed_compare_adjustment = _mm256_set1_epi16(0x8000);
    __m256i threshold_simd = _mm256_set1_epi16((signed)threshold - 0x8000);
    uint8_t* dst_bytes = (uint8_t*)dst;
    uint16_t counters[BITS];

    //Clear out 16-bit counters
    memset(counters, 0, BITS * sizeof(uint16_t));

    //Loop over all input vectors, 255 at a time
    for (int_fast16_t i = 0; i < size; i += 255) {
        uint_fast16_t num_inputs = (i<size-254)? 255: size-i;

        size_t cur_counters = 0;
        for (size_t byte_offset = 0; byte_offset < BYTES; byte_offset += 32) {

            //Call (inline) the function to load half a cache line of input bits from each input hypervector
            __m256i scrambled_counts[8];
            count_half_cacheline_for_255_input_hypervectors_avx2(xs + i, byte_offset, num_inputs, scrambled_counts);

            //Unscramble the counters
            __m256i out_counts[8];
            unscramble_byte_counters_avx2(scrambled_counts, out_counts);

            //Expand the 8-bit counters into 16-bits, and add them to the running counters
            for (int_fast8_t out_i = 0; out_i < 8; out_i++) {
                __m256i increment0 = _mm256_cvtepu8_epi16(((__m128i*)&out_counts[out_i])[0]);
                __m256i increment1 = _mm256_cvtepu8_epi16(((__m128i*)&out_counts[out_i])[1]);
                _mm256_storeu_si256((__m256i*)(&counters[cur_counters + 0]), _mm256_add_epi16(*(__m256i*)(&counters[cur_counters + 0]), increment0));
                _mm256_storeu_si256((__m256i*)(&counters[cur_counters + 16]), _mm256_add_epi16(*(__m256i*)(&counters[cur_counters + 16]), increment1));
                cur_counters += 32;
            }
        }
    }

    //Now do thresholding, and output the final bits
    for (size_t i = 0; i < BITS/16; i++) {
        __m256i adjusted_counters = _mm256_sub_epi16(*(__m256i*)(&counters[i * 16]), signed_compare_adjustment);
        uint64_t maj_words[4];
        *(__m256i *) maj_words = _mm256_cmpgt_epi16(adjusted_counters, threshold_simd);
        uint8_t maj_bytes[2];
        maj_bytes[0] = (uint8_t)_pext_u64(maj_words[0], 0x0001000100010001) | (uint8_t)_pext_u64(maj_words[1], 0x0001000100010001) << 4;
        maj_bytes[1] = (uint8_t)_pext_u64(maj_words[2], 0x0001000100010001) | (uint8_t)_pext_u64(maj_words[3], 0x0001000100010001) << 4;

        *((uint16_t*)(dst_bytes + (i * 2))) = *((uint16_t*)maj_bytes);
    }
}

/// @brief AVX2 implementation of threshold_into using a 1-Byte counter
void threshold_into_byte_avx2(word_t ** xs, uint8_t size, uint8_t threshold, word_t* dst) {
    const __m256i one_twenty_eight = _mm256_set1_epi8((char)128);
    __m256i threshold_simd = _mm256_set1_epi8(((signed char)threshold) - 128);
    uint8_t* dst_bytes = (uint8_t*)dst;

    for (size_t byte_offset = 0; byte_offset < BYTES; byte_offset += 32) {

        //Call (inline) the function to load one cache line of input bits from each input hypervector
        __m256i scrambled_counts[8];
        count_half_cacheline_for_255_input_hypervectors_avx2(xs, byte_offset, size, scrambled_counts);

        //Unscramble the counters
        __m256i out_counts[8];
        unscramble_byte_counters_avx2(scrambled_counts, out_counts);

        //Do the threshold test, and compose out output bits
        __m256i out_bits;
        for (int i=0; i<8; i++) {
            __m256i adjusted_counts = _mm256_sub_epi8(out_counts[i], one_twenty_eight);
            __m256i maj_bits_vec = _mm256_cmpgt_epi8(adjusted_counts, threshold_simd);
            __mmask32 maj_bits = _mm256_movemask_epi8(maj_bits_vec);
            ((uint32_t*)&out_bits)[i] = maj_bits;
        }

        //Store the results
        _mm256_storeu_si256((__m256i*)(dst_bytes + byte_offset), out_bits);
    }
}

/// @brief AVX2 implementation of threshold_into using a 4-Byte counter
void threshold_into_32bit_avx2(word_t ** xs, uint32_t size, uint32_t threshold, word_t* dst) {
    const __m256i signed_compare_adjustment = _mm256_set1_epi32(0x80000000);
    __m256i threshold_simd = _mm256_set1_epi32((signed)threshold - 0x80000000);
    uint8_t* dst_bytes = (uint8_t*)dst;
    uint32_t counters[BITS];

    //Clear out the counters
    memset(counters, 0, BITS * sizeof(uint32_t));

    //Loop over all input vectors, 255 at a time
    for (size_t i = 0; i < size; i += 255) {
        size_t num_inputs = (i<size-254)? 255: size-i;

        size_t cur_counters = 0;
        for (size_t byte_offset = 0; byte_offset < BYTES; byte_offset += 32) {

            //Call (inline) the function to load half a cache line of input bits from each input hypervector
            __m256i scrambled_counts[8];
            count_half_cacheline_for_255_input_hypervectors_avx2(xs + i, byte_offset, num_inputs, scrambled_counts);

            //Unscramble the counters
            __m256i out_counts[8];
            unscramble_byte_counters_avx2(scrambled_counts, out_counts);

            //Expand the 8-bit counters into 32-bits, and add them to the running counters
            for (int_fast8_t out_i = 0; out_i < 8; out_i++) {
                __m128i converted0 = _mm_set1_epi64(((__m64*)&out_counts[out_i])[0]);
                __m128i converted1 = _mm_set1_epi64(((__m64*)&out_counts[out_i])[1]);
                __m128i converted2 = _mm_set1_epi64(((__m64*)&out_counts[out_i])[2]);
                __m128i converted3 = _mm_set1_epi64(((__m64*)&out_counts[out_i])[3]);

                __m256i increment0 = _mm256_cvtepu8_epi32((__m128i) converted0);
                __m256i increment1 = _mm256_cvtepu8_epi32((__m128i) converted1);
                __m256i increment2 = _mm256_cvtepu8_epi32((__m128i) converted2);
                __m256i increment3 = _mm256_cvtepu8_epi32((__m128i) converted3);
                _mm256_storeu_si256((__m256i*)(&counters[cur_counters + 0]), _mm256_add_epi32(*(__m256i*)(&counters[cur_counters + 0]), increment0));
                _mm256_storeu_si256((__m256i*)(&counters[cur_counters + 8]), _mm256_add_epi32(*(__m256i*)(&counters[cur_counters + 8]), increment1));
                _mm256_storeu_si256((__m256i*)(&counters[cur_counters + 16]), _mm256_add_epi32(*(__m256i*)(&counters[cur_counters + 16]), increment2));
                _mm256_storeu_si256((__m256i*)(&counters[cur_counters + 24]), _mm256_add_epi32(*(__m256i*)(&counters[cur_counters + 24]), increment3));
                cur_counters += 32;
            }
        }
    }

    //Now do thresholding, and output the final bits
    for (size_t i = 0; i < BITS/8; i++) {
        __m256i adjusted_counters = _mm256_sub_epi32(*(__m256i*)(&counters[i * 8]), signed_compare_adjustment);
        uint64_t maj_words[4];
        *(__m256i *) maj_words = _mm256_cmpgt_epi32(adjusted_counters, threshold_simd);
        uint8_t maj_bytes;
        maj_bytes = (uint8_t)_pext_u64(maj_words[0], 0x0000000100000001) |
            (uint8_t)_pext_u64(maj_words[1], 0x0000000100000001) << 2 |
            (uint8_t)_pext_u64(maj_words[2], 0x0000000100000001) << 4 |
            (uint8_t)_pext_u64(maj_words[3], 0x0000000100000001) << 6;

        *((uint16_t*)(dst_bytes + i)) = maj_bytes;
    }
}

/// @brief Sets each result bit high if there are more than threshold 1 bits in the corresponding bit of the input vectors
/// @param xs array of `size` input vectors
/// @param size number of input vectors in xs
/// @param threshold threshold to count against
/// @param dst the hypervector to write the results into
void threshold_into_avx2(word_t ** xs, size_t size, size_t threshold, word_t* dst) {
    switch ((size*(size-1))/2 + threshold) {
        case 0: logic_threshold_into_avx2<1, 0>(xs, dst); return;
        case 1: logic_threshold_into_avx2<2, 0>(xs, dst); return;
        case 2: logic_threshold_into_avx2<2, 1>(xs, dst); return;
        case 3: logic_threshold_into_avx2<3, 0>(xs, dst); return;
        case 4: logic_threshold_into_avx2<3, 1>(xs, dst); return;
        case 5: logic_threshold_into_avx2<3, 2>(xs, dst); return;
        case 6: logic_threshold_into_avx2<4, 0>(xs, dst); return;
        case 7: logic_threshold_into_avx2<4, 1>(xs, dst); return;
        case 8: logic_threshold_into_avx2<4, 2>(xs, dst); return;
        case 9: logic_threshold_into_avx2<4, 3>(xs, dst); return;
        case 10: logic_threshold_into_avx2<5, 0>(xs, dst); return;
        case 11: logic_threshold_into_avx2<5, 1>(xs, dst); return;
        case 12: logic_threshold_into_avx2<5, 2>(xs, dst); return;
        case 13: logic_threshold_into_avx2<5, 3>(xs, dst); return;
        case 14: logic_threshold_into_avx2<5, 4>(xs, dst); return;
        case 15: logic_threshold_into_avx2<6, 0>(xs, dst); return;
        case 16: logic_threshold_into_avx2<6, 1>(xs, dst); return;
        case 17: logic_threshold_into_avx2<6, 2>(xs, dst); return;
        case 18: logic_threshold_into_avx2<6, 3>(xs, dst); return;
        case 19: logic_threshold_into_avx2<6, 4>(xs, dst); return;
        case 20: logic_threshold_into_avx2<6, 5>(xs, dst); return;
        case 21: logic_threshold_into_avx2<7, 0>(xs, dst); return;
        case 22: logic_threshold_into_avx2<7, 1>(xs, dst); return;
        case 23: logic_threshold_into_avx2<7, 2>(xs, dst); return;
        case 24: logic_threshold_into_avx2<7, 3>(xs, dst); return;
        case 25: logic_threshold_into_avx2<7, 4>(xs, dst); return;
        case 26: logic_threshold_into_avx2<7, 5>(xs, dst); return;
        case 27: logic_threshold_into_avx2<7, 6>(xs, dst); return;
        case 28: logic_threshold_into_avx2<8, 0>(xs, dst); return;
        case 29: logic_threshold_into_avx2<8, 1>(xs, dst); return;
        case 30: logic_threshold_into_avx2<8, 2>(xs, dst); return;
        case 31: logic_threshold_into_avx2<8, 3>(xs, dst); return;
        case 32: logic_threshold_into_avx2<8, 4>(xs, dst); return;
        case 33: logic_threshold_into_avx2<8, 5>(xs, dst); return;
        case 34: logic_threshold_into_avx2<8, 6>(xs, dst); return;
        case 35: logic_threshold_into_avx2<8, 7>(xs, dst); return;
        case 36: logic_threshold_into_avx2<9, 0>(xs, dst); return;
        case 37: logic_threshold_into_avx2<9, 1>(xs, dst); return;
        case 38: logic_threshold_into_avx2<9, 2>(xs, dst); return;
        case 39: logic_threshold_into_avx2<9, 3>(xs, dst); return;
        case 40: logic_threshold_into_avx2<9, 4>(xs, dst); return;
        case 41: logic_threshold_into_avx2<9, 5>(xs, dst); return;
        case 42: logic_threshold_into_avx2<9, 6>(xs, dst); return;
        case 43: logic_threshold_into_avx2<9, 7>(xs, dst); return;
        case 44: logic_threshold_into_avx2<9, 8>(xs, dst); return;
        case 45: logic_threshold_into_avx2<10, 0>(xs, dst); return;
        case 46: logic_threshold_into_avx2<10, 1>(xs, dst); return;
        case 47: logic_threshold_into_avx2<10, 2>(xs, dst); return;
        case 48: logic_threshold_into_avx2<10, 3>(xs, dst); return;
        case 49: logic_threshold_into_avx2<10, 4>(xs, dst); return;
        case 50: logic_threshold_into_avx2<10, 5>(xs, dst); return;
        case 51: logic_threshold_into_avx2<10, 6>(xs, dst); return;
        case 52: logic_threshold_into_avx2<10, 7>(xs, dst); return;
        case 53: logic_threshold_into_avx2<10, 8>(xs, dst); return;
        case 54: logic_threshold_into_avx2<10, 9>(xs, dst); return;
        case 55: logic_threshold_into_avx2<11, 0>(xs, dst); return;
        case 56: logic_threshold_into_avx2<11, 1>(xs, dst); return;
        case 57: logic_threshold_into_avx2<11, 2>(xs, dst); return;
        case 58: logic_threshold_into_avx2<11, 3>(xs, dst); return;
        case 59: logic_threshold_into_avx2<11, 4>(xs, dst); return;
        case 60: logic_threshold_into_avx2<11, 5>(xs, dst); return;
        case 61: logic_threshold_into_avx2<11, 6>(xs, dst); return;
        case 62: logic_threshold_into_avx2<11, 7>(xs, dst); return;
        case 63: logic_threshold_into_avx2<11, 8>(xs, dst); return;
        case 64: logic_threshold_into_avx2<11, 9>(xs, dst); return;
        case 65: logic_threshold_into_avx2<11, 10>(xs, dst); return;
    }
    if (size < 256) { threshold_into_byte_avx2(xs, size, threshold, dst); return; }
    if (size < 65536) { threshold_into_short_avx2(xs, size, threshold, dst); return; }
    threshold_into_32bit_avx2(xs, size, threshold, dst);
}
#endif //__AVX2__

#if __AVX512BW__
#define threshold_into threshold_into_avx512
#elif __AVX2__
#define threshold_into threshold_into_avx2
#else
#define threshold_into threshold_into_reference<uint32_t>
#endif //#if __AVX512BW__

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
