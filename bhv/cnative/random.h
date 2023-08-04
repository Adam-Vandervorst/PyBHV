void rand_into_reference(word_t *x) {
    for (word_iter_t i = 0; i < WORDS; ++i) {
        x[i] = rng();
    }
}

#ifdef __AESNI__
__m256i aes_state = _mm256_setzero_si256();
__m256i increment = _mm256_set_epi8(0x2f, 0x2b, 0x29, 0x25, 0x1f, 0x1d, 0x17, 0x13,
                                    0x11, 0x0D, 0x0B, 0x07, 0x05, 0x03, 0x02, 0x01,
                                    0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
                                    0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c);

void rand_into_aes(word_t *x) {
    for (word_iter_t i = 0; i < WORDS; i += 8) {
        aes_state += increment;

        __m256i penultimate = _mm256_aesenc_epi128(aes_state, increment);

        _mm256_storeu_si256((__m256i *) (x + i), _mm256_aesenc_epi128(penultimate, increment));
        _mm256_storeu_si256((__m256i *) (x + i + 4), _mm256_aesdec_epi128(penultimate, increment));
    }
}
#endif

#ifdef __AVX2__
avx2_pcg32_random_t avx2_key = {
        .state = {_mm256_set_epi64x(0xb5f380a45f908741, 0x88b545898d45385d, 0xd81c7fe764f8966c, 0x44a9a3b6b119e7bc),
                  _mm256_set_epi64x(0x3cb6e04dc22f629, 0x727947debc931183, 0xfbfa8fdcff91891f, 0xb9384fd8f34c0f49)},
        .inc = {_mm256_set_epi64x(0xbf2de0670ac3d03e, 0x98c40c0dc94e71e, 0xf3565f35a8c61d00, 0xd3c83e29b30df640),
                _mm256_set_epi64x(0x14b7f6e4c89630fa, 0x37cc7b0347694551, 0x4a052322d95d485b, 0x10f3ade77a26e15e)},
        .pcg32_mult_l =  _mm256_set1_epi64x(UINT64_C(0x5851f42d4c957f2d) & 0xffffffffu),
        .pcg32_mult_h = _mm256_set1_epi64x(UINT64_C(0x5851f42d4c957f2d) >> 32)};

void rand_into_avx2(word_t *x) {
    for (word_iter_t i = 0; i < WORDS; i += 4) {
        _mm256_storeu_si256((__m256i *) (x + i), avx2_pcg32_random_r(&avx2_key));
    }
}
#endif //__AVX2__

#if __AVX512BW__
avx512_pcg32_random_t avx512_narrow_key = {
    .state = _mm512_set_epi64(0xb5f380a45f908741, 0x88b545898d45385d, 0xd81c7fe764f8966c, 0x44a9a3b6b119e7bc, 0x3cb6e04dc22f629, 0x727947debc931183, 0xfbfa8fdcff91891f, 0xb9384fd8f34c0f49),
    .inc = _mm512_set_epi64(0xbf2de0670ac3d03e, 0x98c40c0dc94e71e, 0xf3565f35a8c61d00, 0xd3c83e29b30df640, 0x14b7f6e4c89630fa, 0x37cc7b0347694551, 0x4a052322d95d485b, 0x10f3ade77a26e15e),
      .multiplier =  _mm512_set1_epi64(0x5851f42d4c957f2d)};

avx512bis_pcg32_random_t avx512_key = {
    .state = {_mm512_set_epi64(0xb5f380a45f908741, 0x88b545898d45385d, 0xd81c7fe764f8966c, 0x44a9a3b6b119e7bc, 0x3cb6e04dc22f629, 0x727947debc931183, 0xfbfa8fdcff91891f, 0xb9384fd8f34c0f49),
              _mm512_set_epi64(0xe4253e998046cdfb, 0x78a622340a6ad250, 0x5e414281f13fd909, 0x3015456ade10a4d0, 0x7294fe41ba737ee9, 0x36dc2d779e797897, 0x81228ea9c9bb25a2, 0xfbfca70842e57746)},
    .inc = {_mm512_set_epi64(0xbf2de0670ac3d03e, 0x98c40c0dc94e71e, 0xf3565f35a8c61d00, 0xd3c83e29b30df640, 0x14b7f6e4c89630fa, 0x37cc7b0347694551, 0x4a052322d95d485b, 0x10f3ade77a26e15e),
            _mm512_set_epi64(0x5e3cf9dbf6635b3c, 0x2a580d00dc0e34cd, 0xb2b1c52ab1c72ca6, 0x4a683d7ad57caba0, 0x76b85fc2d899c649, 0xf28e80cc844192ff, 0x40a357e9b7739d1e, 0xeb8aa949b57f75de)},
      .multiplier = _mm512_set1_epi64(0x5851f42d4c957f2d)};

void rand_into_avx512(word_t * x) {
    for (word_iter_t i = 0; i < WORDS; i += 8) {
        _mm512_storeu_si512((__m512i*)(x + i), avx512bis_pcg32_random_r(&avx512_key));
    }
}
#endif

#if __AVX512BW__
#define rand_into rand_into_avx512
#elif __AVX2__
#define rand_into rand_into_avx2
#else
#define rand_into rand_into_reference
#endif

void rand2_into_reference(word_t *target, int8_t pow) {
    if (pow == 0)
        return rand_into_reference(target);

    for (word_iter_t i = 0; i < WORDS; ++i) {
        word_t w = rng();
        for (int8_t p = 0; p < std::abs(pow); ++p) {
            if (pow > 0)
                w &= rng();
            else
                w |= rng();
        }
        target[i] = w;
    }
}

#define rand2_into rand2_into_reference

void random_into_reference(word_t *x, float_t p) {
    std::bernoulli_distribution gen(p);

    for (word_iter_t i = 0; i < WORDS; ++i) {
        word_t word = 0;
        for (bit_word_iter_t bit_id = 0; bit_id < BITS_PER_WORD; ++bit_id) {
            if (gen(rng))
                word |= 1UL << bit_id;
        }
        x[i] = word;
    }
}

// Note This could have an AVX-512 implementation with 512-bit float-level log and floor, and probably and equivalent to generate_canonical
template<bool additive>
void sparse_random_switch_into(word_t *x, float_t prob, word_t *target) {
    double inv_log_not_prob = 1. / log(1 - prob);
    size_t skip_count = floor(log(generate_canonical<float_t, 23>(rng)) * inv_log_not_prob);

    for (word_iter_t i = 0; i < WORDS; ++i) {
        word_t word = x[i];
        while (skip_count < BITS_PER_WORD) {
            if constexpr (additive)
                word |= 1UL << skip_count;
            else
                word &= ~(1UL << skip_count);
            skip_count += floor(log(generate_canonical<float_t, 23>(rng)) * inv_log_not_prob);
        }
        skip_count -= BITS_PER_WORD;
        target[i] = word;
    }
}

uint64_t instruction_upto(float target, uint8_t *to, float *remaining, float threshold = 1e-4) {
    uint8_t depth = 0;
    uint64_t res = 0;
    float frac = target;
    float delta;
    float correction;

    do {
        delta = frac - (1.f / (float) (2 << depth));

        if (delta > 0) {
            res |= 1ULL << depth;
            frac = delta;
        }

        depth += 1;
        correction = (1. - target) / (1. - (target + delta)) - 1.;
    } while (abs(correction) > threshold);

    *to = depth - 1;
    *remaining = correction;
    return res;
}

#ifdef __AVX2__

void random_into_tree_sparse_avx2(word_t *x, float p) {
    constexpr float sparse_faster_threshold = .002;

    if (p < sparse_faster_threshold)
        return sparse_random_switch_into<true>(ZERO, p, x);
    else if (p > (1.f - sparse_faster_threshold))
        return sparse_random_switch_into<false>(ONE, 1.f - p, x);

    uint8_t to;
    float correction;
    uint64_t instr = instruction_upto(p, &to, &correction, sparse_faster_threshold);

    for (word_iter_t word_id = 0; word_id < WORDS; word_id += 4) {
        __m256i chunk = avx2_pcg32_random_r(&avx2_key);

        for (uint8_t i = to - 1; i < to; --i) {
            if ((instr & (1ULL << i)) >> i)
                chunk = _mm256_or_si256(chunk, avx2_pcg32_random_r(&avx2_key));
            else
                chunk = _mm256_and_si256(chunk, avx2_pcg32_random_r(&avx2_key));
        }

        _mm256_storeu_si256((__m256i *) (x + word_id), chunk);
    }

    if (correction == 0.)
        return;
    else if (correction > 0.)
        return sparse_random_switch_into<true>(x, correction, x);
    else if (correction < 0.)
        return sparse_random_switch_into<false>(x, -correction, x);
}
#endif //__AVX2__

int8_t ternary_instruction(float af, uint8_t* instr, uint8_t *to, float threshold=1e-6) {
    if (af <= 0.) return -2;
    if (af >= 1.) return -3;

    float da = af - (1.f / (float) (2 << (2*(*to))));

    if (abs(da) <= threshold) return -1;

    if (da > 0) af = da;

    float db = af - (1.f / (float) (2 << (2*(*to) + 1)));

    if (db > 0) af = db;

    if (abs(db) > threshold) {
        if (da > 0) {
            if (db > 0) instr[*to] = 0;
            else instr[*to] = 1;
        } else {
            if (db > 0) instr[*to] = 2;
            else instr[*to] = 3;
        }
        *to += 1;
        return ternary_instruction(af, instr, to, threshold);
    }

    return da > 0;
}

#if __AVX512BW__
void random_into_ternary_tree_avx512(word_t *x, float_t p) {
    if (p < 1.f/128.f)
        return sparse_random_switch_into<true>(ZERO, p, x);
    else if (p > 127.f/128.f)
        return sparse_random_switch_into<false>(ONE, 1.f - p, x);

    uint8_t buffer [24];
    uint8_t to = 0;
    int8_t finalizer = ternary_instruction(p, buffer, &to, 5e-4);

    for (word_iter_t word_id = 0; word_id < WORDS; word_id += 8) {
        __m512i chunk = avx512bis_pcg32_random_r(&avx512_key);

        switch (finalizer) {
            case 1: chunk = _mm512_or_si512(avx512bis_pcg32_random_r(&avx512_key), chunk); break;
            case 0: chunk = _mm512_and_si512(avx512bis_pcg32_random_r(&avx512_key), chunk); break;
            case -1: break;
            case -2: case -3: assert(false);
        }

        for (int i = (int)to - 1; i >= 0; --i) switch (buffer[i]) {
            case 0: chunk = _mm512_ternarylogic_epi64(avx512bis_pcg32_random_r(&avx512_key),
                                                      avx512bis_pcg32_random_r(&avx512_key),
                                                      chunk, 0b11111110); break;
            case 1: chunk = _mm512_ternarylogic_epi64(avx512bis_pcg32_random_r(&avx512_key),
                                                      avx512bis_pcg32_random_r(&avx512_key),
                                                      chunk, 0b11111000); break;
            case 2: chunk = _mm512_ternarylogic_epi64(avx512bis_pcg32_random_r(&avx512_key),
                                                      avx512bis_pcg32_random_r(&avx512_key),
                                                      chunk, 0b11100000); break;
            case 3: chunk = _mm512_ternarylogic_epi64(avx512bis_pcg32_random_r(&avx512_key),
                                                      avx512bis_pcg32_random_r(&avx512_key),
                                                      chunk, 0b10000000); break;
        }

        _mm512_storeu_si512((__m512i *) (x + word_id), chunk);
    }
}
#endif

#if __AVX512BW__
#define random_into random_into_ternary_tree_avx512
#elif __AVX2__
#define random_into random_into_tree_sparse_avx2
#else
#define random_into random_into_reference
#endif //#if __AVX512BW__


word_t *rand() {
    word_t *x = empty();
    rand_into(x);
    return x;
}

word_t *random(float_t p) {
    word_t *x = empty();
    random_into(x, p);
    return x;
}
