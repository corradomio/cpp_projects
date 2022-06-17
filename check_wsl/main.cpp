#include <iostream>
#include <math.h>
#include <vector>
#include <tbb/parallel_for_each.h>


int main() {
    std::vector<int> v;
    tbb::parallel_for_each(v.cbegin(), v.cend(), [&](int i){
        //std::cout << i << std::endl;
    });

    return 0;
}
