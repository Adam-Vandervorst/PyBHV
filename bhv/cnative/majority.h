#if __AVX512BW__
/// @brief AVX-512 implementation of Decision-Tree-Majority algorithm
/// @note Optimal for use in the N=7 to N=89 regime.  After that, thresholding is faster
template <uint8_t size>
void logic_majority_into_avx512(word_t ** xs, word_t* dst) {
    constexpr uint8_t half = size/2;
    __m512i grid [size/2 + 1][size/2 + 1];

    for (word_iter_t word_id = 0; word_id < WORDS; word_id += 8) {

        word_t* x = xs[size - 1];
        grid[half][half] = _mm512_loadu_si512(&(x[word_id]));

        for (uint8_t i = 0; i < half; ++i) {
            x = xs[size - i - 2];
            __m512i chunk = _mm512_loadu_si512(&(x[word_id]));

            grid[half - i - 1][half] = grid[half - i][half] & chunk;
            grid[half][half - i - 1] = grid[half][half - i] | chunk;
        }

        //NOTE: loop terminates when variable wraps after 0
        for (uint8_t i = half - 1; i < half; --i) for (uint8_t j = half - 1; j < half; --j) {
            x = xs[i + j];
            __m512i chunk = _mm512_loadu_si512(&(x[word_id]));

            grid[i][j] = grid[i][j + 1] ^ (chunk & (grid[i][j + 1] ^ grid[i + 1][j]));
        }

        *((__m512i*)&(dst[word_id])) = grid[0][0];
    }
}
#endif //__AVX512BW__

#if !__AVX512BW__
/// @brief AVX-256 implementation of Decision-Tree-Majority algorithm
/// @note Optimal for use in the N=7 to N=79 regime.  After that, thresholding is faster
template <uint8_t size>
void logic_majority_into_avx256(word_t ** xs, word_t* dst) {
    constexpr uint8_t half = size/2;
    __m256i grid [size/2 + 1][size/2 + 1];

    for (word_iter_t word_id = 0; word_id < WORDS; word_id += 4) {
        word_t* x = xs[size - 1];
        grid[half][half] = _mm256_loadu_si256((__m256i*)(x + word_id));

        for (uint8_t i = 0; i < half; ++i) {
            x = xs[size - i - 2];
            __m256i chunk = _mm256_loadu_si256((__m256i*)(x + word_id));

            grid[half - i - 1][half] = grid[half - i][half] & chunk;
            grid[half][half - i - 1] = grid[half][half - i] | chunk;
        }

        //NOTE: loop terminates when variable wraps after 0
        for (uint8_t i = half - 1; i < half; --i) for (uint8_t j = half - 1; j < half; --j) {
            x = xs[i + j];
            __m256i chunk = _mm256_loadu_si256((__m256i*)(x + word_id));

            grid[i][j] = grid[i][j + 1] ^ (chunk & (grid[i][j + 1] ^ grid[i + 1][j]));
        }

        _mm256_storeu_si256((__m256i*)(dst + word_id), grid[0][0]);
    }
}
#endif //!__AVX512BW__

#if __AVX512BW__
    #define logic_majority_into logic_majority_into_avx512
#else
    #define logic_majority_into logic_majority_into_avx256
#endif

//TODO, MAJ-5 for AVX-512

#if __AVX512BW__
/// @brief AVX-512 version of majority3_into
void majority3_into_avx512(word_t * x, word_t * y, word_t * z, word_t * dst) {

    //TODO: It's a bit mysterious why this underperforms the logic_majority_into
    // function on the out-of-cache case, but beats it on the in-cache case.
    // The fetches should dominate everything in the out-of-cache case.

    for (word_iter_t word_id = 0; word_id < WORDS; word_id += 8) {

        word_t words [8];
        words[0] = x[word_id];
        words[1] = x[word_id + 1];
        words[2] = x[word_id + 2];
        words[3] = x[word_id + 3];
        words[4] = x[word_id + 4];
        words[5] = x[word_id + 5];
        words[6] = x[word_id + 6];
        words[7] = x[word_id + 7];
        __m512i _x = *((__m512i *)(words));
        // NOTE: Highly mysterious why _mm512_loadu_si512 performs considerably worse here,
        // but considerably better in the more complex functions.  My guess is that it has
        // something to do with prefetch behavior activated by the other fetches that doesn't
        // come into play with the AVX fetch???
        // __m512i _x = _mm512_loadu_si512(&(x[word_id]));

        words[0] = y[word_id];
        words[1] = y[word_id + 1];
        words[2] = y[word_id + 2];
        words[3] = y[word_id + 3];
        words[4] = y[word_id + 4];
        words[5] = y[word_id + 5];
        words[6] = y[word_id + 6];
        words[7] = y[word_id + 7];
        __m512i _y = *((__m512i *)(words));
        // __m512i _y = _mm512_loadu_si512(&(y[word_id]));

        words[0] = z[word_id];
        words[1] = z[word_id + 1];
        words[2] = z[word_id + 2];
        words[3] = z[word_id + 3];
        words[4] = z[word_id + 4];
        words[5] = z[word_id + 5];
        words[6] = z[word_id + 6];
        words[7] = z[word_id + 7];
        __m512i _z = *((__m512i *)(words));
        // __m512i _z = _mm512_loadu_si512(&(z[word_id]));

        __m512i _x_and_y = _mm512_and_epi64(_x, _y);
        __m512i _x_and_z = _mm512_and_epi64(_x, _z);
        __m512i _y_and_z = _mm512_and_epi64(_y, _z);

        __m512i _result = _mm512_or_epi64(_x_and_y, _x_and_z);
        _result = _mm512_or_epi64(_result, _y_and_z);

        _mm512_storeu_si512(&(dst[word_id]), _result);
        // *((__m512i*)&(dst[word_id])) = _result;
    }
}
#endif //!__AVX512BW__

void majority3_into_avx256(word_t * x, word_t * y, word_t * z, word_t * dst) {
    for (word_iter_t word_id = 0; word_id < WORDS; word_id += 4) {
        __m256i xi = _mm256_loadu_si256((__m256i*)(x + word_id));
        __m256i yi = _mm256_loadu_si256((__m256i*)(y + word_id));
        __m256i zi = _mm256_loadu_si256((__m256i*)(z + word_id));

        __m256i result = ((xi & yi) | (xi & zi) | (yi & zi));

        _mm256_storeu_si256((__m256i*)(dst + word_id), result);
    }
}

/// @brief Straight C implementation of Maj-3
void majority3_into_word(word_t * x, word_t * y, word_t * z, word_t * target) {
    for (word_iter_t i = 0; i < WORDS; ++i) {
        target[i] = ((x[i] & y[i]) | (x[i] & z[i]) | (y[i] & z[i]));
    }
}

#if __AVX512BW__
    #define majority3_into majority3_into_avx512
#else
    #define majority3_into majority3_into_avx256
#endif //#if __AVX512BW__

/// @brief Computes the majority value for each bit among all corresponding bits in the input vectors
/// @param xs array of size input vectors
/// @param size count of input vectors
/// @param dst output vec
void true_majority_into(word_t ** xs, size_t size, word_t *dst) {
    assert(size % 2 == 1 && "true majority must be given an odd number of input hypervectors");
    switch (size) {
        case 1: memcpy(dst, xs[0], BYTES); return;
        case 3: majority3_into(xs[0], xs[1], xs[2], dst); return;
        case 5: logic_majority_into<5>(xs, dst); return; //TODO: MAJ-5 for AVX-512
        case 7: logic_majority_into<7>(xs, dst); return;
        case 9: logic_majority_into<9>(xs, dst); return;
        case 11: logic_majority_into<11>(xs, dst); return;
        case 13: logic_majority_into<13>(xs, dst); return;
        case 15: logic_majority_into<15>(xs, dst); return;
        case 17: logic_majority_into<17>(xs, dst); return;
        case 19: logic_majority_into<19>(xs, dst); return;
        case 21: logic_majority_into<21>(xs, dst); return;
        case 23: logic_majority_into<23>(xs, dst); return;
        case 25: logic_majority_into<25>(xs, dst); return;
        case 27: logic_majority_into<27>(xs, dst); return;
        case 29: logic_majority_into<29>(xs, dst); return;
        case 31: logic_majority_into<31>(xs, dst); return;
        case 33: logic_majority_into<33>(xs, dst); return;
        case 35: logic_majority_into<35>(xs, dst); return;
        case 37: logic_majority_into<37>(xs, dst); return;
        case 39: logic_majority_into<39>(xs, dst); return;
        case 41: logic_majority_into<41>(xs, dst); return;
        case 43: logic_majority_into<43>(xs, dst); return;
        case 45: logic_majority_into<45>(xs, dst); return;
        case 47: logic_majority_into<47>(xs, dst); return;
        case 49: logic_majority_into<49>(xs, dst); return;
        case 51: logic_majority_into<51>(xs, dst); return;
        case 53: logic_majority_into<53>(xs, dst); return;
        case 55: logic_majority_into<55>(xs, dst); return;
        case 57: logic_majority_into<57>(xs, dst); return;
        case 59: logic_majority_into<59>(xs, dst); return;
        case 61: logic_majority_into<61>(xs, dst); return;
        case 63: logic_majority_into<63>(xs, dst); return;
        case 65: logic_majority_into<65>(xs, dst); return;
        case 67: logic_majority_into<67>(xs, dst); return;
        case 69: logic_majority_into<69>(xs, dst); return;
        case 71: logic_majority_into<71>(xs, dst); return;
        case 73: logic_majority_into<73>(xs, dst); return;
        case 75: logic_majority_into<75>(xs, dst); return;
        case 77: logic_majority_into<77>(xs, dst); return;
        case 79: logic_majority_into<79>(xs, dst); return;
        case 81: logic_majority_into<81>(xs, dst); return;
        case 83: logic_majority_into<83>(xs, dst); return;
        case 85: logic_majority_into<85>(xs, dst); return;
        case 87: logic_majority_into<87>(xs, dst); return;
        case 89: logic_majority_into<89>(xs, dst); return;
        default: threshold_into(xs, size, size/2, dst); return;
    }
}

/// @brief Computes the majority value for each bit among all corresponding bits in the input vectors
/// @param xs array of size input vectors
/// @param size count of input vectors
/// @return Return hypervector
word_t* true_majority(word_t ** xs, size_t size) {
    word_t* new_vec = bhv::empty();
    true_majority_into(xs, size, new_vec);
    return new_vec;
}
