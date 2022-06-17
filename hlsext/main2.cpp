#include <vector>
#include <string>
#include <iostream>
#include <random>
#include <stdx/random.h>

#include "hls/util/range.hpp"
#include "hls/util/format.hpp"

using namespace std;
using namespace hls;

void appmain(const vector<string>& args) {
    //util::range_t<int,3> r(0,10);
    //for(int i : r)
    //    cout << util::format("%d", i) << endl;

    srand(0);

    stdx::lcg_random_t r2(0);

    for (int i=0; i<10; ++i)
        printf("%d %d\n", rand(), r2.next_int());



}
