
// IEEE754 Floating Point Access
#if __APPLE__

    //Little Endian 32-bit float
    union ieee754_float {
        float f;
        struct {
            unsigned int mantissa:23;
            unsigned int exponent:8;
            unsigned int negative:1;
        } ieee;
    };

    #define IEEE754_FLOAT_BIAS 0x7f

#else
    #include <ieee754.h>
#endif