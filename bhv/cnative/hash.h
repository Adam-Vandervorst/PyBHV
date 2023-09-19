void rehash_into(word_t *x, word_t *target) {
    TurboSHAKE(512, (uint8_t *) x, BYTES, 0x1F, (uint8_t *) target, BYTES);
}

uint64_t hash(word_t *x) {
    uint8_t buf[8] __attribute__((aligned(8)));
    TurboSHAKE(512, (uint8_t *) x, BYTES, 0x1F, buf, 8);
    return *((uint64_t*)buf);
}
