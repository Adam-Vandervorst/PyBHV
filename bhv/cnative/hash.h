void rehash_into(word_t *x, word_t *target) {
    TurboSHAKE(512, (uint8_t *) x, BYTES, 0x1F, (uint8_t *) target, BYTES);
}
