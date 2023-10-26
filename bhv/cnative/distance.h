/// @brief Count the number of set bits in the vector
/// @note This implementation is within 30% of AVX-2 and AVX-512 on Ice-Lake
bit_iter_t active_reference(word_t *x) {
    bit_iter_t total = 0;
    for (word_iter_t i = 0; i < WORDS; ++i)
        total += std::popcount(x[i]);
    return total;
}

#ifdef __AVX2__

void carry_save_adder(__m256i &h, __m256i &l, __m256i a, __m256i b, __m256i c) {
    __m256i u = _mm256_xor_si256(a, b);
    h = _mm256_or_si256(_mm256_and_si256(a, b), _mm256_and_si256(u, c));
    l = _mm256_xor_si256(u, c);
}

__m256i count(__m256i v) {
    __m256i lookup = _mm256_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3,
                                      1, 2, 2, 3, 2, 3, 3, 4,
                                      0, 1, 1, 2, 1, 2, 2, 3,
                                      1, 2, 2, 3, 2, 3, 3, 4);
    __m256i low_mask = _mm256_set1_epi8(0x0f);
    __m256i lo = _mm256_and_si256(v, low_mask);
    __m256i hi = _mm256_and_si256(_mm256_srli_epi32(v, 4), low_mask);
    __m256i popcnt1 = _mm256_shuffle_epi8(lookup, lo);
    __m256i popcnt2 = _mm256_shuffle_epi8(lookup, hi);
    __m256i total = _mm256_add_epi8(popcnt1, popcnt2);
    return _mm256_sad_epu8(total, _mm256_setzero_si256());
}

/// @brief Count the number of set bits in the vector using an expanded AVX2 adder
/// @note This follows https://github.com/JeWaVe/hamming_rs
bit_iter_t active_adder_avx2(word_t *x) {
    __m256i *vec_x = (__m256i *)x;
    __m256i total = _mm256_setzero_si256();
    __m256i ones = _mm256_setzero_si256();
    __m256i twos = _mm256_setzero_si256();
    __m256i fours = _mm256_setzero_si256();
    __m256i eights = _mm256_setzero_si256();
    __m256i sixteens = _mm256_setzero_si256();
    __m256i twos_a = _mm256_setzero_si256();
    __m256i fours_a = _mm256_setzero_si256();
    __m256i eights_a = _mm256_setzero_si256();
    __m256i twos_b = _mm256_setzero_si256();
    __m256i fours_b = _mm256_setzero_si256();
    __m256i eights_b = _mm256_setzero_si256();
    for (word_iter_t i = 0; i < BITS/256; i += 16) {
        carry_save_adder(twos_a, ones, ones, _mm256_loadu_si256(vec_x + i), _mm256_loadu_si256(vec_x + i + 1));
        carry_save_adder(twos_b, ones, ones, _mm256_loadu_si256(vec_x + i + 2), _mm256_loadu_si256(vec_x + i + 3));
        carry_save_adder(fours_a, twos, twos, twos_a, twos_b);
        carry_save_adder(twos_a, ones, ones, _mm256_loadu_si256(vec_x + i + 4), _mm256_loadu_si256(vec_x + i + 5));
        carry_save_adder(twos_b, ones, ones, _mm256_loadu_si256(vec_x + i + 6), _mm256_loadu_si256(vec_x + i + 7));
        carry_save_adder(fours_b, twos, twos, twos_a, twos_b);
        carry_save_adder(eights_a, fours, fours, fours_a, fours_b);
        carry_save_adder(twos_a, ones, ones, _mm256_loadu_si256(vec_x + i + 8), _mm256_loadu_si256(vec_x + i + 9));
        carry_save_adder(twos_b, ones, ones, _mm256_loadu_si256(vec_x + i + 10), _mm256_loadu_si256(vec_x + i + 11));
        carry_save_adder(fours_a, twos, twos, twos_a, twos_b);
        carry_save_adder(twos_a, ones, ones, _mm256_loadu_si256(vec_x + i + 12), _mm256_loadu_si256(vec_x + i + 13));
        carry_save_adder(twos_b, ones, ones, _mm256_loadu_si256(vec_x + i + 14), _mm256_loadu_si256(vec_x + i + 15));
        carry_save_adder(fours_b, twos, twos, twos_a, twos_b);
        carry_save_adder(eights_b, fours, fours, fours_a, fours_b);
        carry_save_adder(sixteens, eights, eights, eights_a, eights_b);
        total = _mm256_add_epi64(total, count(sixteens));
    }
    // final reduce
    total = _mm256_slli_epi64(total, 4);
    total = _mm256_add_epi64(total, _mm256_slli_epi64(count(eights), 3));
    total = _mm256_add_epi64(total, _mm256_slli_epi64(count(fours), 2));
    total = _mm256_add_epi64(total, _mm256_slli_epi64(count(twos), 1));
    total = _mm256_add_epi64(total, count(ones));
    return (_mm256_extract_epi64(total, 0)
            + _mm256_extract_epi64(total, 1)
            + _mm256_extract_epi64(total, 2)
            + _mm256_extract_epi64(total, 3));
}
#endif //__AVX2__

#if __AVX512BW__
/// @brief Count the number of set bits in the vector using vector popcnt (AVX-512 BITALG)
bit_iter_t active_avx512(word_t *x) {
    __m512i total = _mm512_set1_epi64(0);

    for (word_iter_t i = 0; i < WORDS; i += 8) {
        __m512i v = _mm512_loadu_si512((__m512i *)(x + i));
        __m512i cnts = _mm512_popcnt_epi64(v);
        total = _mm512_add_epi64(total, cnts);
    }

#if true // TODO figure out when reduce_add is available
    return _mm512_reduce_add_epi64(total);
#else
    uint64_t a [8];
    _mm512_storeu_si512((__m512i *)a, totals);
    return a[0] + a[1] + a[2] + a[3] + a[4] + a[5] + a[6] + a[7];
#endif
}
#endif

#if __AVX512BW__
#define active active_avx512
#elif __AVX2__
#if BITS >= 4096
#define active active_adder_avx2
#else
#define active active_reference
#endif
#else
#define active active_reference
#endif

/// @brief The hamming distance between two vectors, this is equivalent to active(xor(x, y)) but faster.
bit_iter_t hamming_reference(word_t *x, word_t *y) {
    bit_iter_t total = 0;
    for (word_iter_t i = 0; i < WORDS; ++i) {
        total += __builtin_popcountl(x[i] ^ y[i]);
    }
    return total;
}

#ifdef __AVX2__

/// @brief The hamming distance between two vectors using an expanded AVX2 adder
/// @note This follows https://github.com/JeWaVe/hamming_rs
uint64_t hamming_adder_avx2(word_t *x, word_t *y) {
    __m256i *vec_x = (__m256i *)x;
    __m256i *vec_y = (__m256i *)y;
    __m256i total = _mm256_setzero_si256();
    __m256i ones = _mm256_setzero_si256();
    __m256i twos = _mm256_setzero_si256();
    __m256i fours = _mm256_setzero_si256();
    __m256i eights = _mm256_setzero_si256();
    __m256i sixteens = _mm256_setzero_si256();
    __m256i twos_a = _mm256_setzero_si256();
    __m256i fours_a = _mm256_setzero_si256();
    __m256i eights_a = _mm256_setzero_si256();
    __m256i twos_b = _mm256_setzero_si256();
    __m256i fours_b = _mm256_setzero_si256();
    __m256i eights_b = _mm256_setzero_si256();
    for (word_iter_t i = 0; i < BITS/256; i += 16) {
        carry_save_adder(twos_a, ones, ones,
            _mm256_xor_si256(_mm256_loadu_si256(vec_x + i), _mm256_loadu_si256(vec_y + i)),
            _mm256_xor_si256(_mm256_loadu_si256(vec_x + i + 1), _mm256_loadu_si256(vec_y + i + 1))
        );
        carry_save_adder(twos_b, ones, ones,
            _mm256_xor_si256(_mm256_loadu_si256(vec_x + i + 2), _mm256_loadu_si256(vec_y + i + 2)),
            _mm256_xor_si256(_mm256_loadu_si256(vec_x + i + 3), _mm256_loadu_si256(vec_y + i + 3))
        );
        carry_save_adder(fours_a, twos, twos, twos_a, twos_b);
        carry_save_adder(twos_a, ones, ones,
            _mm256_xor_si256(_mm256_loadu_si256(vec_x + i + 4), _mm256_loadu_si256(vec_y + i + 4)),
            _mm256_xor_si256(_mm256_loadu_si256(vec_x + i + 5), _mm256_loadu_si256(vec_y + i + 5))
        );
        carry_save_adder(twos_b, ones, ones,
            _mm256_xor_si256(_mm256_loadu_si256(vec_x + i + 6), _mm256_loadu_si256(vec_y + i + 6)),
            _mm256_xor_si256(_mm256_loadu_si256(vec_x + i + 7), _mm256_loadu_si256(vec_y + i + 7))
        );
        carry_save_adder(fours_b, twos, twos, twos_a, twos_b);
        carry_save_adder(eights_a, fours, fours, fours_a, fours_b);
        carry_save_adder(twos_a, ones, ones,
            _mm256_xor_si256(_mm256_loadu_si256(vec_x + i + 8), _mm256_loadu_si256(vec_y + i + 8)),
            _mm256_xor_si256(_mm256_loadu_si256(vec_x + i + 9), _mm256_loadu_si256(vec_y + i + 9))
        );
        carry_save_adder(twos_b, ones, ones,
            _mm256_xor_si256(_mm256_loadu_si256(vec_x + i + 10), _mm256_loadu_si256(vec_y + i + 10)),
            _mm256_xor_si256(_mm256_loadu_si256(vec_x + i + 11), _mm256_loadu_si256(vec_y + i + 11))
        );
        carry_save_adder(fours_a, twos, twos, twos_a, twos_b);
        carry_save_adder(twos_a, ones, ones,
            _mm256_xor_si256(_mm256_loadu_si256(vec_x + i + 12), _mm256_loadu_si256(vec_y + i + 12)),
            _mm256_xor_si256(_mm256_loadu_si256(vec_x + i + 13), _mm256_loadu_si256(vec_y + i + 13))
        );
        carry_save_adder(twos_b, ones, ones,
            _mm256_xor_si256(_mm256_loadu_si256(vec_x + i + 14), _mm256_loadu_si256(vec_y + i + 14)),
            _mm256_xor_si256(_mm256_loadu_si256(vec_x + i + 15), _mm256_loadu_si256(vec_y + i + 15))
        );
        carry_save_adder(fours_b, twos, twos, twos_a, twos_b);
        carry_save_adder(eights_b, fours, fours, fours_a, fours_b);
        carry_save_adder(sixteens, eights, eights, eights_a, eights_b);
        total = _mm256_add_epi64(total, count(sixteens));
    }
    // final reduce
    total = _mm256_slli_epi64(total, 4);
    total = _mm256_add_epi64(total, _mm256_slli_epi64(count(eights), 3));
    total = _mm256_add_epi64(total, _mm256_slli_epi64(count(fours), 2));
    total = _mm256_add_epi64(total, _mm256_slli_epi64(count(twos), 1));
    total = _mm256_add_epi64(total, count(ones));
    return (_mm256_extract_epi64(total, 0)
        + _mm256_extract_epi64(total, 1)
        + _mm256_extract_epi64(total, 2)
        + _mm256_extract_epi64(total, 3));
}
#endif //__AVX2__

#if __AVX512BW__
/// @brief The hamming distance between two vectors using vector popcnt (AVX-512 BITALG)
bit_iter_t hamming_avx512(word_t *x, word_t *y) {
    __m512i total = _mm512_set1_epi64(0);

    for (word_iter_t i = 0; i < WORDS; i += 8) {
        __m512i vec_x = _mm512_loadu_si512((__m512i *)(x + i));
        __m512i vec_y = _mm512_loadu_si512((__m512i *)(y + i));
        __m512i d = _mm512_xor_si512(vec_x, vec_y);
        __m512i cnts = _mm512_popcnt_epi64(d);
        total = _mm512_add_epi64(total, cnts);
    }

#if true // TODO figure out when reduce_add is available
    return _mm512_reduce_add_epi64(total);
#else
    uint64_t a [8];
    _mm512_storeu_si512((__m512i *)a, totals);
    return a[0] + a[1] + a[2] + a[3] + a[4] + a[5] + a[6] + a[7];
#endif
}
#endif

#if __AVX512BW__
#define hamming hamming_avx512
#elif __AVX2__
#if BITS >= 4096
#define hamming hamming_adder_avx2
#else
#define hamming hamming_reference
#endif
#else
#define hamming hamming_reference
#endif
