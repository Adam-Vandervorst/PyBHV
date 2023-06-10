#ifndef BHV_CONSTANTS_H
#define BHV_CONSTANTS_H


using word_t = uint64_t;
using word_iter_t = uint8_t;
using byte_iter_t = uint16_t;
using bit_iter_t = uint16_t;
using bit_word_iter_t = uint8_t;

constexpr bit_word_iter_t BITS_PER_WORD = 64;
constexpr bit_iter_t BITS = 8192;

constexpr byte_iter_t BYTES = BITS/8;
constexpr word_iter_t WORDS = BITS/BITS_PER_WORD;

#endif //BHV_CONSTANTS_H
