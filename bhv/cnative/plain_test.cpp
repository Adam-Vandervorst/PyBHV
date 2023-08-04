#include <iostream>
#include <chrono>

#include "core.h"

using namespace std;

int main() {
    constexpr unsigned long N = 201;

    // burn some cycles to get the OS's attention
    volatile uint64_t x = 0x7834d688d8827099ULL;
    for (size_t i = 0; i < 50000000; ++i)
        x = x + (x % 7);

    auto t0 = chrono::high_resolution_clock::now();

    word_t *rs[N];
    for (size_t i = 0; i < N; ++i)
        rs[i] = bhv::rand();

    auto t1 = chrono::high_resolution_clock::now();
    cout << "rand " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << endl;


    word_t *ps[N];
    for (size_t i = 0; i < N; ++i) {
        ps[i] = bhv::empty();
        bhv::roll_word_bits_into(rs[i], 42, ps[i]);
    }

    auto t2 = chrono::high_resolution_clock::now();
    cout << "new permute " << chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count() << endl;

    for (size_t i = 0; i < N; ++i) {
        word_t tmp [WORDS];
        bhv::roll_word_bits_into(ps[i], -42, tmp);
        assert(bhv::eq(rs[i], tmp));
    }

    auto t3 = chrono::high_resolution_clock::now();
    cout << "rpermute eq " << chrono::duration_cast<chrono::nanoseconds>(t3 - t2).count() << endl;

    word_t *m = bhv::true_majority(rs, N);

    auto t4 = chrono::high_resolution_clock::now();
    cout << "majority " << chrono::duration_cast<chrono::nanoseconds>(t4 - t3).count() << endl;

#if false
    word_t * ds[N];
    for (size_t i = 0; i < N; ++i) {
        word_t * d = bhv::empty();
        bhv::xor_into(rs[i], m, d);
        ds[i] = d;
    }

    auto t5 = chrono::high_resolution_clock::now();
    cout << "xor " << chrono::duration_cast<chrono::nanoseconds>(t5 - t4).count() << endl;

    unsigned long qs[N];
    for (size_t i = 0; i < N; ++i)
        qs[i] = bhv::active(ds[i]);

    auto t6 = chrono::high_resolution_clock::now();
    cout << "active " << chrono::duration_cast<chrono::nanoseconds>(t6 - t5).count() << endl;
#else
    unsigned long total = 0;
    for (size_t i = 0; i < N; ++i)
        total += bhv::hamming(rs[i], m);

    auto t5 = chrono::high_resolution_clock::now();
    cout << "hamming " << chrono::duration_cast<chrono::nanoseconds>(t5 - t4).count() << endl;
#endif

    cout << ((double) total / (double) N) << endl;
    return 0;
}