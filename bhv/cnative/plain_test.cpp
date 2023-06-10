#include <iostream>
#include <chrono>

#include "packed.h"

using namespace std;

int main() {
    unsigned long N = 201;


    auto t0 = chrono::high_resolution_clock::now();

    word_t * rs[N];
    for (size_t i = 0; i < N; ++i)
        rs[i] = bhv::rand();

    auto t1 = chrono::high_resolution_clock::now();
    cout << "rand " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << endl;

    word_t * m = bhv::true_majority(rs, N);

    auto t2 = chrono::high_resolution_clock::now();
    cout << "majority " << chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count() << endl;

#if false
    word_t * ds[N];
    for (size_t i = 0; i < N; ++i) {
        word_t * d = bhv::empty();
        bhv::xor_into(rs[i], m, d);
        ds[i] = d;
    }

    auto t3 = chrono::high_resolution_clock::now();
    cout << "xor " << chrono::duration_cast<chrono::nanoseconds>(t3 - t2).count() << endl;

    unsigned long qs[N];
    for (size_t i = 0; i < N; ++i)
        qs[i] = bhv::active(ds[i]);

    auto t4 = chrono::high_resolution_clock::now();
    cout << "active " << chrono::duration_cast<chrono::nanoseconds>(t4 - t3).count() << endl;
#else
    unsigned long qs[N];
    for (size_t i = 0; i < N; ++i)
        qs[i] = bhv::hamming(rs[i], m);

    auto t3 = chrono::high_resolution_clock::now();
    cout << "hamming " << chrono::duration_cast<chrono::nanoseconds>(t3 - t2).count() << endl;
#endif

    unsigned long total = 0;
    for (size_t i = 0; i < N; ++i)
        total += qs[i];

    cout << ((double)total/(double )N) << endl;
    return 0;
}