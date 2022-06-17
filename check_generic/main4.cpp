//
// Created by Corrado Mio (Local) on 02/07/2021.
//
#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <stdx/random.h>

void appmain4(const std::vector<std::string>& args){

    //for(auto& s : args)
    //    std::cout << s << std::endl;

    printf("---\n");
    ::srand(110);
    for (int i=0; i<10; ++i)
        printf("1 %d\n", rand()%100);

    printf("---\n");
    ::srand(110);
    for (int i=0; i<10; ++i)
        printf("2 %d\n", rand()%100);

    printf("---\n");
    stdx::lcg_random_t r2(110);
    for (int i=0; i<10; ++i)
        printf("3 %d\n", r2.next_int()%100);

    printf("---\n");
    stdx::lcg_random_t r3(110);
    for (int i=0; i<10; ++i)
        printf("4 %d\n", r3.next_int()%100);

    printf("---\n");
    return;
}
