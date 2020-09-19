//
// Created by Corrado Mio on 19/09/2020.
//

#ifndef CHECK_HLSEXT_RANDOM_H
#define CHECK_HLSEXT_RANDOM_H

#include <random>

namespace stdx {

    struct random_t {
        std::random_device rd;
        std::mt19937 mt;
        std::uniform_real_distribution<double> dist;

        random_t(): mt(rd()), dist(0, 1){ }

        double next_double() { return dist(mt); }
        double next_double(double a, double b){ return a + (b-a)*dist(mt); }

        int next_int() { return mt(); }
        int next_int(int b) { return mt()%b; }
        int next_int(int a, int b) { return a + mt()%(b-a); }
    };

}

#endif //CHECK_HLSEXT_RANDOM_H
