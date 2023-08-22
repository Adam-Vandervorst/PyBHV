/// @brief Straight C implementation of Maj-3
void majority3_into_reference(word_t *x, word_t *y, word_t *z, word_t *target) {
    for (word_iter_t i = 0; i < WORDS; ++i) {
        target[i] = ((x[i] & y[i]) | (x[i] & z[i]) | (y[i] & z[i]));
    }
}

void true_majority_into_reference(word_t **xs, size_t size, word_t *target) {
    assert(size % 2 == 1 && "true majority must be given an odd number of input hypervectors");
    switch (size) {
        case 1: memcpy(target, xs[0], BYTES); return;
        case 3: majority3_into_reference(xs[0], xs[1], xs[2], target); return;
        default: threshold_into(xs, size, size/2, target); return;
    }
}

#if __AVX512BW__

/// @brief AVX-512 implementation of Decision-Tree-Majority algorithm
/// @note Optimal for use in the N=7 to N=89 regime.  After that, thresholding is faster
template <uint8_t size>
void logic_majority_into_avx512(word_t ** xs, word_t* target) {
    constexpr uint8_t half = size/2;
    __m512i grid [size/2 + 1][size/2 + 1];

    for (word_iter_t word_id = 0; word_id < WORDS; word_id += 8) {

        word_t* x = xs[size - 1];
        grid[half][half] = _mm512_loadu_si512((__m512i*)(x + word_id));

        for (uint8_t i = 0; i < half; ++i) {
            x = xs[size - i - 2];
            __m512i chunk = _mm512_loadu_si512((__m512i*)(x + word_id));

            grid[half - i - 1][half] = grid[half - i][half] & chunk;
            grid[half][half - i - 1] = grid[half][half - i] | chunk;
        }

        //NOTE: loop terminates when variable wraps after 0
        for (uint8_t i = half - 1; i < half; --i) for (uint8_t j = half - 1; j < half; --j) {
            x = xs[i + j];
            __m512i chunk = _mm512_loadu_si512((__m512i*)(x + word_id));

            grid[i][j] = _mm512_ternarylogic_epi64(chunk, grid[i + 1][j], grid[i][j + 1], 0xca); // select
        }

        _mm512_storeu_si512((__m512i*)(target + word_id), grid[0][0]);
    }
}

/// @brief AVX-512 AND-OR version of majority3_into
void majority3_into_avx512(word_t * x, word_t * y, word_t * z, word_t * target) {
    for (word_iter_t word_id = 0; word_id < WORDS; word_id += 8) {
        __m512i xi = _mm512_loadu_si512((__m512i*)(x + word_id));
        __m512i yi = _mm512_loadu_si512((__m512i*)(y + word_id));
        __m512i zi = _mm512_loadu_si512((__m512i*)(z + word_id));

        __m512i result = ((xi & yi) | (xi & zi) | (yi & zi));

        _mm512_storeu_si512((__m512i*)(target + word_id), result);
    }
}

/// @brief AVX-512 TERNARY version of majority3_into
/// @note On GCC 13 majority3_into_avx512 gets compiled into two ternary instructions, this uses only one
void majority3_into_ternary_avx512(word_t * x, word_t * y, word_t * z, word_t * target) {
    __m512i *x_vec = (__m512i *)x;
    __m512i *y_vec = (__m512i *)y;
    __m512i *z_vec = (__m512i *)z;
    __m512i *target_vec = (__m512i *)target;

    for (word_iter_t i = 0; i < BITS/512; ++i) {
        _mm512_storeu_si512(target_vec + i,
                            _mm512_ternarylogic_epi64(_mm512_loadu_si512(x_vec + i),
                                                      _mm512_loadu_si512(y_vec + i),
                                                      _mm512_loadu_si512(z_vec + i), 0xe8));
    }
}

/// @brief Computes the majority value for each bit among all corresponding bits in the input vectors
/// @param xs array of size input vectors
/// @param size count of input vectors
/// @param target output vec
void true_majority_into_avx512(word_t **xs, size_t size, word_t *target) {
    assert(size % 2 == 1 && "true majority must be given an odd number of input hypervectors");
    switch (size) {
        case 1: memcpy(target, xs[0], BYTES); return;
        case 3: majority3_into_ternary_avx512(xs[0], xs[1], xs[2], target); return;
        case 5: logic_majority_into_avx512<5>(xs, target); return;
        case 7: logic_majority_into_avx512<7>(xs, target); return;
        case 9: logic_majority_into_avx512<9>(xs, target); return;
        case 11: logic_majority_into_avx512<11>(xs, target); return;
        case 13: logic_majority_into_avx512<13>(xs, target); return;
        case 15: logic_majority_into_avx512<15>(xs, target); return;
        case 17: logic_majority_into_avx512<17>(xs, target); return;
        case 19: logic_majority_into_avx512<19>(xs, target); return;
        default: threshold_into_avx512(xs, size, size/2, target); return;
    }
}
#endif //!__AVX512BW__

#ifdef __AVX2__
/// @brief AVX-256 implementation of Decision-Tree-Majority algorithm
/// @note Optimal for use in the N=7 to N=79 regime.  After that, thresholding is faster
template<uint8_t size>
void logic_majority_into_avx2(word_t **xs, word_t *target) {
    constexpr uint8_t half = size / 2;
    __m256i grid[size / 2 + 1][size / 2 + 1];

    for (word_iter_t word_id = 0; word_id < WORDS; word_id += 4) {
        word_t *x = xs[size - 1];
        grid[half][half] = _mm256_loadu_si256((__m256i *) (x + word_id));

        for (uint8_t i = 0; i < half; ++i) {
            x = xs[size - i - 2];
            __m256i chunk = _mm256_loadu_si256((__m256i *) (x + word_id));

            grid[half - i - 1][half] = grid[half - i][half] & chunk;
            grid[half][half - i - 1] = grid[half][half - i] | chunk;
        }

        //NOTE: loop terminates when variable wraps after 0
        for (uint8_t i = half - 1; i < half; --i)
            for (uint8_t j = half - 1; j < half; --j) {
                x = xs[i + j];
                __m256i chunk = _mm256_loadu_si256((__m256i *) (x + word_id));

                grid[i][j] = grid[i][j + 1] ^ (chunk & (grid[i][j + 1] ^ grid[i + 1][j])); // select
            }

        _mm256_storeu_si256((__m256i *) (target + word_id), grid[0][0]);
    }
}

/// @brief AVX-2 version of majority3_into
void majority3_into_avx2(word_t *x, word_t *y, word_t *z, word_t *target) {
    for (word_iter_t word_id = 0; word_id < WORDS; word_id += 4) {
        __m256i xi = _mm256_loadu_si256((__m256i *) (x + word_id));
        __m256i yi = _mm256_loadu_si256((__m256i *) (y + word_id));
        __m256i zi = _mm256_loadu_si256((__m256i *) (z + word_id));

        __m256i result = ((xi & yi) | (xi & zi) | (yi & zi));

        _mm256_storeu_si256((__m256i *) (target + word_id), result);
    }
}

/// @brief Computes the majority value for each bit among all corresponding bits in the input vectors
/// @param xs array of size input vectors
/// @param size count of input vectors
/// @param target output vec
void true_majority_into_avx2(word_t **xs, size_t size, word_t *target) {
    assert(size % 2 == 1 && "true majority must be given an odd number of input hypervectors");
    switch (size) {
        case 1: memcpy(target, xs[0], BYTES); return;
        case 3: majority3_into_avx2(xs[0], xs[1], xs[2], target); return;
        case 5: logic_majority_into_avx2<5>(xs, target); return;
        case 7: logic_majority_into_avx2<7>(xs, target); return;
        case 9: logic_majority_into_avx2<9>(xs, target); return;
        case 11: logic_majority_into_avx2<11>(xs, target); return;
        case 13: logic_majority_into_avx2<13>(xs, target); return;
        case 15: logic_majority_into_avx2<15>(xs, target); return;
        default: threshold_into_avx2(xs, size, size/2, target); return;
    }
}
#endif //__AVX2__

#if __AVX512BW__
#define majority3_into majority3_into_ternary_avx512
#define true_majority_into true_majority_into_avx512
#elif __AVX2__
#define majority3_into majority3_into_avx2
#define true_majority_into true_majority_into_avx2
#else
#define majority3_into majority3_into_reference
#define true_majority_into true_majority_into_reference
#endif

/// @brief Computes the majority value for each bit among all corresponding bits in the input vectors
/// @param xs array of size input vectors
/// @param size count of input vectors
/// @return Return hypervector
word_t *true_majority(word_t **xs, size_t size) {
    word_t *new_vec = bhv::empty();
    true_majority_into(xs, size, new_vec);
    return new_vec;
}
