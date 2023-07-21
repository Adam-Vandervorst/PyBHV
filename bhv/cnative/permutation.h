uint8_t permute_byte_bits(uint8_t x, uint64_t p) {
    uint64_t w = _pdep_u64(x, 0x0101010101010101);
    uint64_t res = (uint64_t)_mm_shuffle_pi8(_mm_cvtsi64_m64(w), _mm_cvtsi64_m64(p));
    return _pext_u64(res, 0x0101010101010101);
}

uint64_t bits_of_byte_permutation_invert(uint64_t p) {
    uint64_t r = 0;

    for (uint64_t i = 0; i < 8; ++i)
        r |= (i << ((p >> i*8) & 0x07)*8);

    return r;
}

uint64_t rand_bits_of_byte_permutation(uint32_t seed) {
    std::minstd_rand0 perm_rng(seed);

    uint8_t p [8] = {0, 1, 2, 3, 4, 5, 6, 7};

    std::shuffle(p, p + 8, perm_rng);

    return *((uint64_t*)p);
}

void permute_words_into(word_t * x, word_iter_t* word_permutation, word_t * target) {
    for (word_iter_t i = 0; i < WORDS; ++i) {
        target[i] = x[word_permutation[i]];
    }
}

void inverse_permute_words_into(word_t * x, word_iter_t* word_permutation, word_t * target) {
    for (word_iter_t i = 0; i < WORDS; ++i) {
        target[word_permutation[i]] = x[i];
    }
}

word_iter_t* rand_word_permutation(uint32_t seed) {
    std::minstd_rand0 perm_rng(seed);

    auto p = (word_iter_t *) malloc(sizeof(word_iter_t)*WORDS);

    for (word_iter_t i = 0; i < WORDS; ++i)
        p[i] = i;

    std::shuffle(p, p + WORDS, perm_rng);

    return p;
}

void permute_bits_of_bytes_into(word_t * x, int32_t perm_id, word_t * target) {
    if (perm_id == 0) memcpy(target, x, BYTES);
    else {
        uint8_t* x_bytes = (uint8_t*)x;
        uint8_t* target_bytes = (uint8_t*)target;

        uint64_t byte_perm = rand_bits_of_byte_permutation(perm_id);
        if (perm_id < 0) byte_perm = bits_of_byte_permutation_invert(byte_perm);

        for (byte_iter_t i = 0; i < BYTES; ++i)
            target[i] = permute_byte_bits(x[i], byte_perm);
    }
}

void permute_into(word_t * x, int32_t perm, word_t * target) {
    if (perm == 0) memcpy(target, x, BYTES);
    else if (perm > 0) permute_words_into(x, rand_word_permutation(perm), target);
    else inverse_permute_words_into(x, rand_word_permutation(-perm), target);
}
