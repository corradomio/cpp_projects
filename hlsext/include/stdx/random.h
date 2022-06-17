//
// Created by Corrado Mio on 19/09/2020.
//

#ifndef CHECK_HLSEXT_RANDOM_H
#define CHECK_HLSEXT_RANDOM_H

#include <random>
#include <inttypes.h>

namespace stdx {

    struct random_t {
        std::random_device rd;
        std::mt19937 mt;
        std::uniform_real_distribution<double> dist;

        random_t(): mt(rd()), dist(0, 1){ }

        random_t& seed(int seed) {
            mt.seed(seed);
            return *this;
        }

        double next_double() { return dist(mt); }
        double next_double(double a, double b){ return a + (b-a)*dist(mt); }

        int next_int() { return mt(); }
        int next_int(int b) { return mt()%b; }
        int next_int(int a, int b) { return a + mt()%(b-a); }
    };

    // https://en.wikipedia.org/wiki/Linear_congruential_generator

    struct lcg_random_t {
        long long modulus;        // mm
        long long multiplier;     // a
        long long increment;      // c
        long long value;

        // X[n+1] = (a X[n] + c) mod m

        lcg_random_t() {
            this->init();
        }

        lcg_random_t(int seed) {
            this->init();
            this->seed(seed);
        }

        void init() {
            modulus = 0x7FFFFFFF;
            multiplier = 1103515145;
            increment = 16807;
            value = 0;
        }

        lcg_random_t& seed(int seed) {
            value = seed;
            return *this;
        }

        double next_double() { return (0.+next_int())/modulus; }
        double next_double(double a, double b){ return a + (b-a)*next_double(); }

        int next_int() {
            return value = (multiplier*value + increment) % modulus;
        }
        int next_int(int b) { return next_int()%b; }
        int next_int(int a, int b) { return a + next_int()%(b-a); }
    };

}

#endif //CHECK_HLSEXT_RANDOM_H
