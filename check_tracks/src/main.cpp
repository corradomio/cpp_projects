#include <string>
#include <iostream>

#include "../include/dworld.h"
#include <stdx/keys.h>
#include <tbb/parallel_for_each.h>

using namespace hls::khalifa::summer;


std::string grid_fname(int side, int interval) {
    std::string fname = string_format(R"(D:\Projects.github\cpp_projects\check_tracks\data\tracksgrid_%d_%d_3months.bin)", side, interval);
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
        load_grid(dworld, grid_fname(5, 0));
    }

    std::vector<int> sides{5,10,20,50,100};
    std::vector<int> intervals{1,5,10,15,30,60};

    for(int side : sides) {
        tbb::parallel_for_each(intervals.begin(), intervals.end(), [&](int interval) {
            DiscreteWorld dworld;
            load_grid(dworld, grid_fname(side, interval));
        });
    }

}

void create_load_grid() {
    std::string fname = grid_fname(100, 60);

    create_grid(100, 60, fname);

    DiscreteWorld dworld;
    load_grid(dworld, fname);
}

typedef std::unordered_set<std::string> encounters_t;

void save_encounters(const DiscreteWorld& dworld) {

    const std::vector<std::string>& ids = dworld.get_ids();

    for(const std::string& id : ids) {
        std::cout << "-- " << id << " --" << std::endl;

        std::map<int, encounters_set_t> encs = dworld.get_encounters(id);

        std::vector<int> tlist = stdx::keys(encs, true);

        for(int t : tlist) {
            if (encs[t].eids.size() > 1)
                std::cout << "  " << to_simple_string(dworld.to_timestamp(t)) << ": " << encs[t].eids.size() << std::endl;
        }

    }

}

void print_encounters() {
    DiscreteWorld dworld;
    dworld.load(grid_fname(100, 60));

    save_encounters(dworld);
}


int main() {
    //pfor();
    //create_load_grid();

    //crete_grids();
    
    //load_grids();

    print_encounters();

    return 0;
}
