#include "core.h"
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <chrono>
#include <iostream>


using namespace std;


word_t** load_pbm(FILE* file, size_t* n_elements) {
    char header[2];
    fread(header, 1, 2, file);
    assert(header[0] == 'P' && header[1] == '4');

    int dimension, n;
    fscanf(file, "%d %d", &dimension, &n);
    assert(dimension == BITS);

    *n_elements = n;
    static vector<uint8_t *> hvs = std::vector<uint8_t *>(n);

    for (int i = 0; i < n; ++i) {
        hvs[i] = (uint8_t *)bhv::empty();
        fread(hvs[i], 1, BYTES, file);
    }

    return (word_t **)(hvs.data());
}

void step_into(word_t *hv, word_t *target) {
    word_t nbs [8][WORDS];
    word_t* nbs_lookup [8] = {nbs[0], nbs[1], nbs[2], nbs[3],
                              nbs[4], nbs[5], nbs[6], nbs[7]};

    bhv::roll_word_bits_into(hv, -1, nbs[0]);
    bhv::roll_word_bits_into(hv, 1, nbs[1]);
    bhv::roll_words_into(hv, -1, nbs[2]);
    bhv::roll_words_into(hv, 1, nbs[3]);

    bhv::roll_words_into(nbs[0], 1, nbs[4]);
    bhv::roll_words_into(nbs[0], -1, nbs[5]);
    bhv::roll_words_into(nbs[1], 1, nbs[6]);
    bhv::roll_words_into(nbs[1], -1, nbs[7]);

    word_t alive [WORDS];
    word_t dead [WORDS];

    bhv::window_into(nbs_lookup, 8, 2, 3, alive);
    bhv::window_into(nbs_lookup, 8, 3, 3, dead);

    bhv::select_into(hv, alive, dead, target);
}

int main() {
    FILE* file = fopen("gol1000.pbm", "rb");
    if (file == nullptr) {
        perror("Failed to open file");
        return 1;
    }

    size_t n;
    word_t** hvs = load_pbm(file, &n);
    assert(n == 1001);

    fclose(file);

    word_t* petri_dish_hv = hvs[0];

    for (size_t i = 0; i < 30; ++i)
        step_into(petri_dish_hv, petri_dish_hv);

    cout << bhv::eq(petri_dish_hv, hvs[30]) << endl;

    auto t0 = chrono::high_resolution_clock::now();

    for (size_t i = 0; i < 1000000; ++i)
        step_into(petri_dish_hv, petri_dish_hv);

    auto t1 = chrono::high_resolution_clock::now();

    cout << (1000000.)/((double)chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count()/1e9) << " fps" << endl;

    for (size_t i = 0; i < n; ++i)
        free(hvs[i]);

    return 0;
}