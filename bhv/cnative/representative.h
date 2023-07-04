word_t * representative_impl(word_t ** xs, size_t size) {
    word_t * x = zero();

    std::uniform_int_distribution<size_t> gen(0, size - 1);
    for (word_iter_t word_id = 0; word_id < WORDS; ++word_id) {
        word_t word = 0;
        for (bit_word_iter_t bit_id = 0; bit_id < BITS_PER_WORD; ++bit_id) {
            size_t x_id = gen(rng);
            if ((xs[x_id][word_id] >> bit_id) & 1)
                word |=  1UL << bit_id;
        }
        x[word_id] = word;
    }

    return x;
}

word_t * n_representatives_impl(word_t ** xs, size_t size) {
    word_t * x = zero();

    std::uniform_int_distribution<size_t> gen(0, size - 1);
    for (word_iter_t word_id = 0; word_id < WORDS; ++word_id) {
        word_t word = 0;
        for (bit_word_iter_t bit_id = 0; bit_id < BITS_PER_WORD; ++bit_id) {
            size_t x_id = gen(rng);
            word |=  1UL << (xs[x_id][word_id] >> bit_id) & 1;
        }
        x[word_id] = word;
    }

    return x;
}

word_t* representative(word_t ** xs, size_t size) {
    if (size == 0) return rand();
    else if (size == 1) { word_t * r = empty(); memcpy(r, xs[0], BYTES); return r; }
    else if (size == 2) { word_t * r = rand(); select_into(r, xs[0], xs[1], r); return r; }
    else return representative_impl(xs, size);
}
