//
// Created by Corrado Mio on 19/09/2020.
//

#include "dworld.h"
#include <tbb/parallel_for_each.h>

using namespace hls::khalifa::summer;

std::string grid_fname(int side, int interval) {
    std::string fname = stdx::format(R"(D:\Projects.github\cpp_projects\check_tracks\data\tracksgrid_%d_%d_3months.bin)", side, interval);
    return fname;
}


void crete_grids() {

    create_grid(5, 0, grid_fname(5, 0));

    std::vector<int> sides{5,10,20,50,100};
    std::vector<int> intervals{1,5,10,15,30,60};

    for(int side : sides) {
        tbb::parallel_for_each(intervals.begin(), intervals.end(), [&](int interval) {
            std::string fname = grid_fname(side, interval);
            std::cout << fname << std::endl;

            create_grid(side, interval, grid_fname(side, interval));
        });
    }

}

void load_grids() {

    {
        DiscreteWorld dworld;
        dworld.load(grid_fname(5, 0));
    }

    std::vector<int> sides{5,10,20,50,100};
    std::vector<int> intervals{1,5,10,15,30,60};

    for(int side : sides) {
        tbb::parallel_for_each(intervals.begin(), intervals.end(), [&](int interval) {
            DiscreteWorld dworld;
            dworld.load(grid_fname(side, interval));
        });
    }

}

