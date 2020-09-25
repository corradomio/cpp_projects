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

std::vector<std::tuple<int, int>> make_params(bool skip50=false) {
    std::vector<std::tuple<int, int>> params;

    std::vector<int> sides{5,10,20,50,100};
    std::vector<int> intervals{1,5,10,15,30,60};

    if (!skip50)
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
        dworld.dump();
    });

}

// --------------------------------------------------------------------------
// Suspended
// --------------------------------------------------------------------------

void save_time_encounters(int side, int interval) {
    DiscreteWorld dworld;
    dworld.load(grid_fname(side, interval));

    std::string filename = stdx::format(R"(D:\Projects.github\cpp_projects\check_tracks\encounters\by_time\encounters_%d_%d_3months.csv)", side, interval);
    dworld.save_time_encounters(filename);
}

void save_time_encounters() {

    std::vector<std::tuple<int, int>> params = make_params();
    tbb::parallel_for_each(params.cbegin(), params.cend(), [&](const std::tuple<int, int>& p) {
        int side = std::get<0>(p);
        int interval = std::get<1>(p);

        save_time_encounters(side, interval);
    });
}

// --------------------------------------------------------------------------

void save_slot_encounters(int side, int interval) {
    DiscreteWorld dworld;
    dworld.load(grid_fname(side, interval));

    std::string filename = stdx::format(R"(D:\Projects.github\cpp_projects\check_tracks\encounters\by_slot\encounters_%d_%d_3months.csv)", side, interval);
    dworld.save_slot_encounters(filename);
}

void save_slot_encounters() {

    //save_slot_encounters(100, 60);
    //return

    //save_slot_encounters(5, 0);
    //std::vector<int> sides{5,10,20,50,100};
    //std::vector<int> intervals{1,5,10,15,30,60};

    std::vector<std::tuple<int, int>> params = make_params();

    tbb::parallel_for_each(params.begin(), params.end(), [&](const std::tuple<int, int>& p) {
        int side = std::get<0>(p);
        int interval = std::get<1>(p);

        save_slot_encounters(side, interval);
    });
}

