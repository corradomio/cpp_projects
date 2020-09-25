//
// Created by Corrado Mio on 19/09/2020.
//

#include "dworld.h"
#include <tbb/parallel_for_each.h>

using namespace hls::khalifa::summer;

// --------------------------------------------------------------------------

std::string grid_fname(int side, int interval) {
    std::string fname = stdx::format(
    R"(D:\Projects.github\cpp_projects\check_tracks\data\tracksgrid_%d_%d_3months.bin)",
        side, interval);
    return fname;
}


// --------------------------------------------------------------------------

std::vector<std::tuple<int, int>> make_params() {
    std::vector<std::tuple<int, int>> params;

    std::vector<int> sides{5,10,20,50,100};
    std::vector<int> intervals{1,5,10,15,30,60};

    params.emplace_back(5, 0);

    for (int side : sides)
        for (int interval: intervals)
            params.emplace_back(side, interval);

    return params;
}


// --------------------------------------------------------------------------

void crete_grids() {

    std::vector<std::tuple<int, int>> params = make_params();

    tbb::parallel_for_each(params.begin(), params.end(), [&](const std::tuple<int,int>& p) {
        int side = std::get<0>(p);
        int interval = std::get<1>(p);

        create_grid(side, interval, grid_fname(side, interval));
    });

}

void load_grids() {

    std::vector<std::tuple<int, int>> params = make_params();

    tbb::parallel_for_each(params.begin(), params.end(), [&](const std::tuple<int,int>& p) {
        int side = std::get<0>(p);
        int interval = std::get<1>(p);

        DiscreteWorld dworld;
        dworld.load(grid_fname(side, interval));
    });

}

