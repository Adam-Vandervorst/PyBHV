#ifndef BHV_CONSTANTS_H
#define BHV_CONSTANTS_H


using word_t = uint64_t;

#if DIMENSION/64 > 65536
using word_iter_t = uint32_t;
#elif DIMENSION/64 > 256
using word_iter_t = uint16_t;
#else
using word_iter_t = uint8_t;
#endif

#if DIMENSION/8 > 65536
using byte_iter_t = uint32_t;
#elif DIMENSION/8 > 256
using byte_iter_t = uint16_t;
#else
using byte_iter_t = uint8_t;
#endif

#if DIMENSION > 65536
using bit_iter_t = uint32_t;
#else
using bit_iter_t = uint16_t;
#endif

using bit_word_iter_t = uint8_t;

#define BITS_PER_WORD 64
#define BITS DIMENSION

#define BYTES (BITS / 8)
#define WORDS (BITS / BITS_PER_WORD)

#endif //BHV_CONSTANTS_H
