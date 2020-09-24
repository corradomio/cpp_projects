#include <string>
#include <iostream>

//#include <stdx/to_string.h>
#include <tbb/parallel_for_each.h>

#include "dworld.h"
#include "infections.h"
#include "other.h"

using namespace hls::khalifa::summer;


//std::string grid_fname(int side, int interval) {
//    std::string fname = stdx::format(R"(D:\Projects.github\cpp_projects\check_tracks\data\tracksgrid_%d_%d_3months.bin)", side, interval);
//    return fname;
//}

void simulate(const DiscreteWorld& dworld,
              contact_mode cmode, double cmode_prob,
              double iprob) {

    Infections infections;
    infections
        .set_d(2)
        .set_beta(0.1)
        .set_dworld(dworld)
        .set_contact_mode(cmode, cmode_prob)
        .set_infected(iprob);

    infections.propagate();
}


void simulate(int side, int interval) {
    DiscreteWorld dworld;
    dworld.load(grid_fname(side, interval));
    simulate(dworld, none, 1., .05);
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

    std::string fname = stdx::format(R"(D:\Projects.github\cpp_projects\check_tracks\data\tracksgrid_%d_%d_3months.bin)", side, interval);
    std::string oname = stdx::format(R"(D:\Projects.github\cpp_projects\check_tracks\encounters\by_time\encounters_%d_%d_3months.csv)", side, interval);

    dworld.save_time_encounters(oname);
}

void save_time_encounters() {

    //save_time_encounters(100, 60);

    save_time_encounters(5, 0);

    std::vector<int> sides{5,10,20,50,100};
    std::vector<int> intervals{1,5,10,15,30,60};
    for (int side : sides)
        //for (int interval : intervals)
        //    save_time_encounters(side, interval);
        tbb::parallel_for_each(sides.begin(), sides.end(), [&](int interval) {
            save_time_encounters(side, interval);
        });
}

// --------------------------------------------------------------------------

void save_slot_encounters(int side, int interval) {
    DiscreteWorld dworld;
    dworld.load(grid_fname(side, interval));

    std::string fname = stdx::format(R"(D:\Projects.github\cpp_projects\check_tracks\data\tracksgrid_%d_%d_3months.bin)", side, interval);
    std::string oname = stdx::format(R"(D:\Projects.github\cpp_projects\check_tracks\encounters\by_slot\encounters_%d_%d_3months.csv)", side, interval);

    dworld.save_slot_encounters(oname);
}

void save_slot_encounters() {

    //save_slot_encounters(100, 60);

    save_slot_encounters(5, 0);

    std::vector<int> sides{5,10,20,50,100};
    std::vector<int> intervals{1,5,10,15,30,60};
    for (int side : sides)
        //tbb::parallel_for(intervals, [&](int interval) {
        //    save_slot_encounters(side, interval);
        //});
        for (int interval : intervals)
            save_slot_encounters(side, interval);
}


int main() {
    //pfor();
    //create_load_grid();

    //crete_grids();
    
    //load_grids();

    //save_slot_encounters();

    save_time_encounters();

    //simulate();

    //DiscreteWorld dworld;
    //dworld.load(grid_fname(100, 60));


    return 0;
}
