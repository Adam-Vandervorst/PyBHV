g++ benchmark.cpp TurboSHAKEopt/TurboSHAKE.cpp TurboSHAKEopt/KeccakP-1600-opt64.cpp \
  TurboSHAKEAVX512/TurboSHAKE.cpp TurboSHAKEAVX512/KeccakP-1600-AVX512.cpp \
  -O3 -std=c++20 -march=native -Wall -Wpedantic -Wextra -g -ffast-math
./a.out
rm a.out
