#include <vector>
#include <string>
#include<stdio.h>
#include <math.h>

void appmain(const std::vector<std::string>& apps) {
    double x, y;

    for (int i=1; i<256; ++i) {
        x = 1./i;
        y = x*i;
        if (y != 1)
            printf("1/%d -> no (%g)\n", i, abs(1-y));
    }

}
