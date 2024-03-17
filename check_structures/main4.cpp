//
// Created by Corrado Mio on 07/03/2024.
//
#include <iostream>
#include "stdx/float64/vector_op.h"
#include "stdx/float64/matrix_op.h"
#include "stdx/float64/dot_op.h"
#include "stdx/float64/transpose.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <string>

int main() {
    std::random_device r;
    // initialize the seed
    std::default_random_engine e1(r());
    std::uniform_real_distribution unif(0., 1.);

    for (int i=0; i<10; ++i) {
        printf("%d\n", e1());
        // printf("%f\n", unif(e1));
    }

    return 0;
}

int main13()
{
    // Seed with a real random value, if available
    std::random_device r;

    // Choose a random mean between 1 and 6
    std::default_random_engine e1(r());
    std::uniform_int_distribution<int> uniform_dist(1, 6);
    int mean = uniform_dist(e1);
    std::cout << "Randomly-chosen mean: " << mean << '\n';

    // Generate a normal distribution around that mean
    std::seed_seq seed2{r(), r(), r(), r(), r(), r(), r(), r()};
    std::mt19937 e2(seed2);
    std::normal_distribution<> normal_dist(mean, 2);

    std::map<int, int> hist;
    for (int n = 0; n != 10000; ++n)
        ++hist[std::round(normal_dist(e2))];

    std::cout << "Normal distribution around " << mean << ":\n"
              << std::fixed << std::setprecision(1);
    for (auto [x, y] : hist)
        std::cout << std::setw(2) << x << ' ' << std::string(y / 200, '*') << '\n';

    return 0;
}

using namespace stdx::float64;

int main12() {
    matrix_t A = range(5, 3);
    tr_t T = tr(A);

    print(A);
    print(T);

    // matrix_t A = range(5, 3);
    // matrix_t B = range(2, 3);
    //
    // print(A);
    // print(B);
    //
    // matrix_t C = dott(A, B);
    //
    // print(C);

    // matrix_t A = range(3, 5);
    // matrix_t B = range(3, 2);
    //
    // print(A);
    // print(B);
    //
    // matrix_t C = tdot(A, B);
    //
    // print(C);


    // vector_t u = range(5);
    // vector_t v = ones(10);
    // matrix_t m = range(5, 10);
    // vector_t r;
    //
    // r = dot(m, v);
    // r = dot(u, m);

    // matrix_t A = range(3,3);
    // matrix_t B = identity(3);
    //
    // print(A);
    // print(B);
    //
    // matrix_t C = dot(A, B);
    // print(C);

    // matrix_t A = range(5, 3);
    // matrix_t B = range(3, 2);
    //
    // print(A);
    // print(B);

    // vector_t v(5);
    // v = 3;
    return 0;
}

int main11() {
    // vector_t u{10};
    vector_t v = range(10);
    matrix_t m{5,10};

    // u = ones(10);
    // v = range(10);
    // m = range(5, 10);
    //
    // print(u);
    // print(m);
    //
    // v = dot(m, u);
    //
    // print(v);
    // print(vector_t());
    //
    // m = identity(5);
    // print(m);
    // print(dot(m, m));
    //
    // std::cout << dot(u,v) << std::endl;
    //
    // for(auto it=v.begin(); it!=v.end(); ++it) {
    //     std::cout << *it << std::endl;
    // }

    std::cout << "----------" << std::endl;

    m = range(5, 10);

    for (auto it = m.row_begin(1); it != m.row_end(1); ++it) {
        std::cout << *it << std::endl;
    }

    std::cout << "--" << std::endl;

    for (auto it = m.col_begin(1); it != m.col_end(1); ++it) {
        std::cout << *it << std::endl;
    }

    std::cout << "----------" << std::endl;

    m = m.reshape(10,5);

    for (auto it = m.row_begin(1); it != m.row_end(1); ++it) {
        std::cout << *it << std::endl;
    }

    std::cout << "--" << std::endl;

    for (auto it = m.col_begin(1); it != m.col_end(1); ++it) {
        std::cout << *it << std::endl;
    }

    std::cout << "----------" << std::endl;

    return 0;
}