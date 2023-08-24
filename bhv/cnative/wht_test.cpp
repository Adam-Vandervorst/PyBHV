#include <iostream>
#include "core.h"

using namespace std;


int main() {
    auto v1 = bhv::random(.95);
    bhv::rehash_into(v1, v1);

    cout << bhv::active(v1) << " initial af" << endl;

    auto v1_t = bhv::permute(v1, 0);
    bhv::bfwht(v1_t);

    cout << bhv::active(v1_t) << " initial transformed af" << endl;

    cout << bhv::hamming(v1, v1_t) << " |v,v_t|" << endl;

    auto v1_t_t = bhv::permute(v1_t, 0);
    bhv::bfwht(v1_t_t);

    cout << bhv::active(v1_t_t) << " twice transformed af" << endl;

    cout << bhv::hamming(v1, v1_t_t) << " |v,v_t_t|" << endl;
    cout << bhv::hamming(v1_t, v1_t_t) << " |v_t,v_t_t|" << endl;
}
