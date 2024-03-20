//
// Created by Corrado Mio on 20/03/2024.
//
#include "stdx/tprintf.h"
#include "stdx/float64/linalg.h"

using namespace stdx::float64;

int main62() {
    // 1000x500     6s
    // 2000x500     26s
    // 3000x500     50s
    matrix_t M = range(3000,500);

    stdx::tprintf(); printf("start\n");
    printf("frobenius: %f\n", frobenius( dot(M, M, false, true)));
    stdx::tprintf(); printf("done\n");

    return 0;
}


int main61() {
    // MKLVersion mkl_version;
    // mkl_get_version(&mkl_version);
    // stdx::tprintf(); printf("You are using oneMKL %d.%d\n", mkl_version.MajorVersion, mkl_version.UpdateVersion);

    // 1000x500     16s
    // 2000x500     47s
    // 3000x500
    matrix_t M = range(3,3);
    print(M);

    stdx::tprintf(); printf("start\n");
    // matrix_t R1 = dot(M, M, false, true);
    // matrix_t R2 = dot(M, tr(M));
    printf("frobenius: %f\n", frobenius( dot(M, M, false, false)));
    printf("frobenius: %f\n", frobenius( dot(M, M, false, true)));
    printf("frobenius: %f\n", frobenius( dot(M, M, true, false)));
    printf("frobenius: %f\n", frobenius( dot(M, M, true, true)));

    stdx::tprintf(); printf("done\n");

    print(dot(M, M, false, false));
    print(dot(M, M, false, true));
    print(dot(M, M, true, false));
    print(dot(M, M, true, true));

    printf("---\n");

    M = range(4,3);
    print(dot(M, M, false, true));
    print(dot(M, M, true, false));

    return 0;
}