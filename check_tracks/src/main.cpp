#include <string>
#include <iostream>

#include <tbb/parallel_for_each.h>

#include "dworld.h"
#include "infections.h"
#include "other.h"

using namespace hls::khalifa::summer;


// --------------------------------------------------------------------------


void simulate(const DiscreteWorld& dworld,
              int d, double beta,
              contact_mode cmode, double cmode_prob,
              double iprob) {

    Infections infections;
    infections
        .contact_range(d)
        .infection_rate(beta)
        .contact_mode(cmode, cmode_prob)
        .dworld(dworld);

    infections.infected(iprob);

    infections.init();
    infections.propagate();

    std::string filename = stdx::format(R"(D:\Projects.github\cpp_projects\check_tracks\infections\infections_%d_%d_3months.csv)",
                                        dworld.side(), dworld.interval().total_seconds()/60);
    infections.save(filename, time_duration(24, 0, 0));
}


// --------------------------------------------------------------------------

void simulate(int side, int interval) {
    DiscreteWorld dworld;
    dworld.load(grid_fname(side, interval));

    simulate(dworld, 2, 0.01, none, 1., .05);
}

void simulate() {
//    simulate(5, 0);
    simulate(100, 60);
}


// --------------------------------------------------------------------------

//void crete_grids() {
//
//    create_grid(5, 0, grid_fname(5, 0));
//
//    std::vector<int> sides{5,10,20,50,100};
//    std::vector<int> intervals{1,5,10,15,30,60};
//
//    for(int side : sides) {
//        tbb::parallel_for_each(intervals.begin(), intervals.end(), [&](int interval) {
//            std::string fname = grid_fname(side, interval);
//            std::cout << fname << std::endl;
//
//            create_grid(side, interval, grid_fname(side, interval));
//        });
//    }
//
//}

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


int main() {
    crete_grids();
    //load_grids();

    //save_slot_encounters();
    //save_time_encounters();

    //simulate();

    return 0;
}
