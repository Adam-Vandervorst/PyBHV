void roll_words_into(word_t *x, int32_t d, word_t *target) {
    int32_t offset = ((d % WORDS) + WORDS) % WORDS;

    memcpy(target, x + offset, (WORDS - offset) * sizeof(word_t));
    memcpy(target + WORDS - offset, x, offset * sizeof(word_t));
}

void roll_word_bits_into(word_t *x, int32_t d, word_t *target) {
    int32_t offset = d % BITS_PER_WORD;

    for (word_iter_t i = 0; i < WORDS; ++i) {
        target[i] = std::rotl(x[i], offset);
    }
}

uint8_t permute_single_byte_bits(uint8_t x, uint64_t p) {
    uint64_t w = _pdep_u64(x, 0x0101010101010101);
    uint64_t res = (uint64_t) _mm_shuffle_pi8(_mm_cvtsi64_m64(w), _mm_cvtsi64_m64(p));
    return _pext_u64(res, 0x0101010101010101);
}

uint64_t byte_bits_permutation_invert(uint64_t p) {
    uint64_t r = 0;

    for (uint64_t i = 0; i < 8; ++i)
        r |= i << (((p >> (i * 8)) & 0x07) * 8);

    return r;
}

uint64_t rand_byte_bits_permutation(uint32_t seed) {
    std::minstd_rand0 perm_rng(seed);

    uint8_t p[8] = {0, 1, 2, 3, 4, 5, 6, 7};

    std::shuffle(p, p + 8, perm_rng);

    return *((uint64_t *) p);
}

void permute_byte_bits_into(word_t *x, int32_t perm_id, word_t *target) {
    if (perm_id == 0) {
        memcpy(target, x, BYTES);
        return;
    }

    uint8_t *x_bytes = (uint8_t *) x;
    uint8_t *target_bytes = (uint8_t *) target;

    uint64_t byte_perm = rand_byte_bits_permutation(abs(perm_id));
    if (perm_id < 0) byte_perm = byte_bits_permutation_invert(byte_perm);

    for (byte_iter_t i = 0; i < BYTES; ++i)
        target_bytes[i] = permute_single_byte_bits(x_bytes[i], byte_perm);
}

#if __AVX512BW__
uint64_t permute_single_word_bits(uint64_t x, __m512i p) {
    __m512i x_simd = _mm512_set1_epi64(x);
    __mmask64 permuted_bits = _mm512_bitshuffle_epi64_mask(x_simd, p);
    return _cvtmask64_u64(permuted_bits);
}

__m512i word_bits_permutation_invert(__m512i p) {
    uint8_t p_array [64];
    uint8_t r_array [64];
    _mm512_storeu_si512(p_array, p);

    for (uint8_t i = 0; i < 64; ++i)
        r_array[p_array[i] & 0x3f] = i;

    return _mm512_loadu_si512(r_array);
}

__m512i rand_word_bits_permutation(uint32_t seed) {
    std::minstd_rand0 perm_rng(seed);

    uint8_t p [64];
    for (uint8_t i = 0; i < 64; ++i)
        p[i] = i;

    std::shuffle(p, p + 64, perm_rng);

    return _mm512_loadu_si512(p);
}

void permute_word_bits_into(word_t * x, int32_t perm_id, word_t * target) {
    if (perm_id == 0) {memcpy(target, x, BYTES); return;}

    __m512i word_perm = rand_word_bits_permutation(abs(perm_id));
    if (perm_id < 0) word_perm = word_bits_permutation_invert(word_perm);

    for (word_iter_t i = 0; i < WORDS; ++i)
        target[i] = permute_single_word_bits(x[i], word_perm);
}
#endif

template<bool inverse>
void apply_word_permutation_into(word_t *x, word_iter_t *word_permutation, word_t *target) {
    for (word_iter_t i = 0; i < WORDS; ++i) {
        if constexpr (inverse)
            target[word_permutation[i]] = x[i];
        else
            target[i] = x[word_permutation[i]];
    }
}

void rand_word_permutation_into(uint32_t seed, word_iter_t *p) {
    std::minstd_rand0 perm_rng(seed);

    for (word_iter_t i = 0; i < WORDS; ++i)
        p[i] = i;

    std::shuffle(p, p + WORDS, perm_rng);
}

void permute_words_into(word_t *x, int32_t perm, word_t *target) {
    if (perm == 0) {
        memcpy(target, x, BYTES);
        return;
    }

    word_iter_t p[WORDS];
    rand_word_permutation_into(abs(perm), p);

    if (perm > 0) apply_word_permutation_into<false>(x, p, target);
    else apply_word_permutation_into<true>(x, p, target);
}


template<bool inverse>
void apply_byte_permutation_into(word_t *x, byte_iter_t *byte_permutation, word_t *target) {
    uint8_t *x_bytes = (uint8_t *) x;
    uint8_t *target_bytes = (uint8_t *) target;

    for (byte_iter_t i = 0; i < BYTES; ++i) {
        if constexpr (inverse)
            target_bytes[byte_permutation[i]] = x_bytes[i];
        else
            target_bytes[i] = x_bytes[byte_permutation[i]];
    }
}

void rand_byte_permutation_into(uint32_t seed, byte_iter_t *p) {
    std::minstd_rand0 perm_rng(seed);

    for (byte_iter_t i = 0; i < BYTES; ++i)
        p[i] = i;

    std::shuffle(p, p + BYTES, perm_rng);
}

void permute_bytes_into(word_t *x, int32_t perm, word_t *target) {
    if (perm == 0) {
        memcpy(target, x, BYTES);
        return;
    }

    byte_iter_t p[BYTES];
    rand_byte_permutation_into(abs(perm), p);

    if (perm > 0) apply_byte_permutation_into<false>(x, p, target);
    else apply_byte_permutation_into<true>(x, p, target);
}

void permute_into(word_t *x, int32_t perm, word_t *target) {
#if __AVX512BW__
    permute_words_into(x, perm, target);
    permute_word_bits_into(target, perm, target);
#else
    permute_bytes_into(x, perm, target);
    permute_byte_bits_into(target, perm, target);
#endif
}
