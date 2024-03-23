#include <iostream>
#include <chrono>
#include <bits/chrono_io.h>
#include "cuda/cublas.h"


int main11() {
    typedef std::chrono::high_resolution_clock Clock;

    int N=1000;
    int M=500;

    double* m1 = new double[N*M];
    double* m2 = new double[M*N];
    double* m3 = new double[N*N];

    auto t1 = Clock::now();
    for (int i=0,i1=0,i3=0; i<1000; ++i,i1+=M,i3+=N) {
        for (int j=0; j<1000; ++j) {
            double s=0;
            for (int k=0,k2=0; k<500; ++k,k2+=N)
                // s += m1[i*M+k]*m2[k*N+j];
                s += m1[i1+k]*m2[k2+j];

            // m3[i*N+j] = s;
            m3[i3+j] = s;
        }
    }
    // tprint(); std::cout << "done" << std::endl;
    auto t2 = Clock::now();
    std::cout << std::chrono::duration<float, std::milli>(t2-t1) <<  std::endl;

    std::cout << "Hello, World!" << std::endl;
    return 0;
}
