//
// Created by Corrado Mio (Local) on 02/07/2021.
//
#include <iostream>
#include <vector>
#include <string>
#include <random>

void appmain(const std::vector<std::string>& args){

    for(auto& s : args)
        std::cout << s << std::endl;

    ::srand(110);
    for (int i=0; i<10; ++i)
        printf("1 %d\n", rand()%100);

    ::srand(110);
    for (int i=0; i<10; ++i)
        printf("2 %d\n", rand()%100);
}
