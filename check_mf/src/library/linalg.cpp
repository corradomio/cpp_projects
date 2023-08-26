//
// Created by Corrado Mio on 03/08/2023.
//
#include "linalg.h"

namespace hls::linalg {

    array<float> create(int n) {
        array<float> vec(n);
        for (int i=0; i<n; ++i)
            vec[i] = (i+1.f);
        return vec;
    }

    array<float> create(int n, int m) {
        array<float> mat(n, m);

        for (int i=0; i<n; ++i)
            for (int j=0; j<m; ++j)
                mat[i, j] = (i+1.f)*m + (j+1.f);
        return mat;
    }
}

