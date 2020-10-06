#include <iostream>
#include <math.h>
#include <vector>
#include <tbb/parallel_for_each.h>
#include <stdx/ranges.h>


int main() {
    std::vector<int> v;

    stdx::range_t<int> r = stdx::range(100000000);
    v.assign(r.cbegin(), r.cend());

    std::cout << v.size() << std::endl;

    tbb::parallel_for_each(v.cbegin(), v.cend(), [&](int i){
        //std::cout << i << std::endl;
    });

    return 0;
}
