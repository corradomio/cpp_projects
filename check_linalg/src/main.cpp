#include <iostream>
#include "stdx/linalg.h"

int main() {
    std::cout << "Hello, World!" << std::endl;

    stdx::linalg::vector_t<float> v1(100);
    stdx::linalg::vector_t<float> v2(v1);
    stdx::linalg::vector_t<float> v3(v1, true);

    v2[0] = 101;

    v3 = v2;
    v1 = 0;

    stdx::linalg::matrix_t<float> m1(200, 100);
    for (int i=0; i<200; ++i)
        for (int j=0; j<100; ++j)
            m1.at(i,j) = (1+i)*100+(1+j);

    stdx::linalg::matrix_t<float> m2(m1);
    stdx::linalg::matrix_t<float> m3(m1, true);

    // m2[0,0] = 112;

    stdx::linalg::trmat_t mt = stdx::linalg::tr(m1);

    printf("%lld x %lld\n", m3.rows(), m3.cols());
    printf("%lld x %lld\n", mt.rows(), mt.cols());

    m3 = m2;
    // m1 = 0;

    printf("\n");
    for (int i=0; i<5; ++i) {
        for (int j=0; j<5; ++j)
            printf("%3.0f ", mt.at(i,j));
            // printf("%3.0f ", mt[i,j]);
        printf("\n");
    }


    return 0;
}
