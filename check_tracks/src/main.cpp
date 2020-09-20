#include <string>
#include <iostream>

//#include <stdx/to_string.h>
#include <tbb/parallel_for_each.h>

#include "dworld.h"
#include "infections.h"

using namespace hls::khalifa::summer;


std::string grid_fname(int side, int interval) {
    std::string fname = stdx::format(R"(D:\Projects.github\cpp_projects\check_tracks\data\tracksgrid_%d_%d_3months.bin)", side, interval);
    return fname;
}

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
    simulate(100, 60);
}


int main() {
    //pfor();
    //create_load_grid();

    //crete_grids();
    
    //load_grids();

    //save_encounters();

    simulate();


    return 0;
}
