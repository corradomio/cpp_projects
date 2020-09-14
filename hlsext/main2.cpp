#include <vector>
#include <string>
#include <iostream>

#include "hls/util/range.hpp"
#include "hls/util/format.hpp"

using namespace std;
using namespace hls;

void appmain(const vector<string>& args) {
    util::range_t<int,3> r(0,10);
    for(int i : r)
        cout << util::format("%d", i) << endl;
}
