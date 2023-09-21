template <typename N>
N opt_independent(word_t* x, std::function<N (word_t*)> loss) {
    N best_l = loss(x);
    for (bit_iter_t i = 0; i < BITS; ++i) {
        toggle(x, i);
        N l = loss(x);
        if (l > best_l)
            toggle(x, i);
        else
            best_l = l;
    }
    return best_l;
}

template <typename N, double U, double L, uint64_t half_time>
float_t time_based_decay(uint64_t time_step, N loss) {
    double decay_rate = log((U - L)/(U + L))/(double)half_time;
    return L + (U - L)*exp(decay_rate*(double)time_step);
}

template <typename N>
N opt_linear_search(word_t* x, std::function<N (word_t*)> loss,
                    uint64_t max_iter = 10000000,
                    N break_at = std::numeric_limits<N>::min(),
                    std::function<float_t (uint64_t, N)> update = time_based_decay<N, 0.001, 0.0001, 100000>) {
    N best_l = loss(x);
    N l;
    word_t change [WORDS];
    word_t buf [WORDS];
    bool x_best = true;
    float_t delta = update(0, best_l);

    for (uint64_t it = 1; it <= max_iter; ++it) {
        random_into(change, delta);

        if (x_best) {
            xor_into(x, change, buf);
            l = loss(buf);
        } else {
            xor_into(buf, change, x);
            l = loss(x);
        }

        if (l < best_l) {
//            std::cout << "#";
            best_l = l;
            x_best = not x_best;
        } else {
//            std::cout << ".";
        }

        if (l <= break_at) {
            break;
        }

        delta = update(it, l);
    }

    if (not x_best)
        memcpy(x, buf, BYTES);

    return best_l;
}
