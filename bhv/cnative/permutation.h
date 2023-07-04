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

void permute_into(word_t * x, int32_t perm, word_t * target) {
    if (perm == 0) *target = *x;
    else if (perm > 0) permute_words_into(x, rand_word_permutation(perm), target);
    else inverse_permute_words_into(x, rand_word_permutation(-perm), target);
}
