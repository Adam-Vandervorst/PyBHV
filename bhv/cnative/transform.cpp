#include "core.h"
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <chrono>
#include <iostream>


using namespace std;


int main() {
    FILE* file = fopen("gol1024.pbm", "rb");
    if (file == nullptr) {
        perror("Failed to open file");
        return 1;
    }

    size_t n;
    word_t** hvs = bhv::load_pbm(file, &n);
    assert(n == 1024);

    fclose(file);

    auto t0 = chrono::high_resolution_clock::now();

    for (size_t i = 0; i < n; ++i)
        bhv::rehash_into(hvs[i], hvs[i]);

    auto t1 = chrono::high_resolution_clock::now();

    cout << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << endl;

    FILE* ofile = fopen("hash_gol1024.pbm", "wb");

    bhv::save_pbm(ofile, hvs, n);

    fclose(ofile);

    for (size_t i = 0; i < n; ++i)
        free(hvs[i]);

    return 0;
}
