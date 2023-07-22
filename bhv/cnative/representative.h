/// @brief Plain C implementation of representative sampling
void representative_into_reference(word_t **xs, size_t size, word_t *target) {
    std::uniform_int_distribution<size_t> gen(0, size - 1);
    for (word_iter_t word_id = 0; word_id < WORDS; ++word_id) {
        word_t word = 0;
        for (bit_word_iter_t bit_id = 0; bit_id < BITS_PER_WORD; ++bit_id) {
            size_t x_id = gen(rng);
            if ((xs[x_id][word_id] >> bit_id) & 1)
                word |= 1UL << bit_id;
        }
        target[word_id] = word;
    }
}

//word_t * n_representatives_impl(word_t ** xs, size_t size) {
//    word_t * x = zero();
//
//    std::uniform_int_distribution<size_t> gen(0, size - 1);
//    for (word_iter_t word_id = 0; word_id < WORDS; ++word_id) {
//        word_t word = 0;
//        for (bit_word_iter_t bit_id = 0; bit_id < BITS_PER_WORD; ++bit_id) {
//            size_t x_id = gen(rng);
//            word |=  1UL << (xs[x_id][word_id] >> bit_id) & 1;
//        }
//        x[word_id] = word;
//    }
//
//    return x;
//}

/// @brief For each dimension, samples a bit from xs into target
void representative_into(word_t **xs, size_t size, word_t *target) {
    if (size == 0) rand_into(target);
    else if (size == 1) memcpy(target, xs[0], BYTES);
    else if (size == 2) { word_t r[WORDS]; rand_into(r); select_into(r, xs[0], xs[1], target); }
    else representative_into_reference(xs, size, target);
}