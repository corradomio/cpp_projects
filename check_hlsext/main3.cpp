//
// Created by Corrado Mio on 23/09/2020.
//

#include <iostream>
#include "stdx/ranges.h"

int main3() {

    printf("%d\n", sizeof(long long));
    printf("%d\n", sizeof(long));
    printf("%d\n", sizeof(int));
    printf("%d\n", sizeof(size_t));

    printf("--\n", sizeof(size_t));
    auto  r = stdx::range(2,5);

    //for (auto it = r.begin(); it != r.end(); ++it)
    //    std::cout << *it << std::endl;

    for (int i : stdx::range<int>(5))
        std::cout << i << std::endl;

    return 0;
}
