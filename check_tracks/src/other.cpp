//
// Created by Corrado Mio on 19/09/2020.
//

#include <csvstream.h>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <tbb/parallel_for_each.h>
#include "infections.h"
#include <stdx/ranges.h>

using namespace boost::filesystem;
using namespace boost;
using namespace hls::khalifa::summer;

// --------------------------------------------------------------------------
// Parameters
// --------------------------------------------------------------------------

const std::string& DATASET = R"(D:\Dropbox\2_Khalifa\Progetto Summer\Dataset_3months)";

std::string grid_fname(int side, int interval) {
    std::string fname = stdx::format(
    R"(D:\Projects.github\cpp_projects\check_tracks\data\tracksgrid_%d_%d_3months.bin)",
        side, interval);
    return fname;
}

std::vector<std::tuple<int, int>> make_params(bool skip50=false) {
    std::vector<std::tuple<int, int>> params;

    std::vector<int> sides{5,10,20,50,100};
    std::vector<int> intervals{1,5,10,15,30,60};

    if (!skip50) params.emplace_back(5, 0);

    for (int side : sides)
        for (int interval: intervals)
            params.emplace_back(side, interval);

    return params;
}

// --------------------------------------------------------------------------
// Generic functions
// --------------------------------------------------------------------------

void create_grid(int side, int interval, const std::string& filename) {

    std::cout << "create_grid(" << side << "," << interval << ") ..." << std::endl;

    DiscreteWorld dworld(side, interval);

    try {
        int count = 0;

        path p(DATASET);

        // 0,  1          2           3    4          5                  6      7      8           9
        // "","latitude","longitude","V3","altitude","date.Long.format","date","time","person.id","track.id"

        for (directory_entry &de : directory_iterator(p)) {
            if (de.path().extension() != ".csv")
                continue;

            csvstream csvin(de.path().string());

            std::vector<std::string> row;

            while  (csvin >> row) {
                count += 1;

                //if (count%100000 == 0)
                //    std::cout << "    " << count << std::endl;

                user_t user = lexical_cast<user_t>(row[8]);
                double latitude  = lexical_cast<double>(row[1]);
                double longitude = lexical_cast<double>(row[2]);

                date date = from_string(row[6]);
                time_duration duration = duration_from_string(row[7]);
                ptime timestamp(date, duration);

                dworld.add(user, latitude, longitude, timestamp);
            }
        }
        dworld.done();
        std::cout << "    " << count << std::endl;

        std::cout << "save in(" << filename << ")" << std::endl;
        dworld.save(filename);
        dworld.dump();
        std::cout << std::endl;
    }
    catch(std::exception& e) {
        std::cout << e.what() << std::endl;
    }

}

void create_grids() {

    std::vector<std::tuple<int, int>> params = make_params();

    tbb::parallel_for_each(params.begin(), params.end(), [&](const std::tuple<int,int>& p) {
        int side = std::get<0>(p);
        int interval = std::get<1>(p);

        create_grid(side, interval, grid_fname(side, interval));
    });

}

void create_grid_test(){
    create_grid(100, 60, grid_fname(100, 60));
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
// IO
// --------------------------------------------------------------------------

void save_encounters(int side, int interval) {
    std::string filename;
    DiscreteWorld dworld;

    dworld.load(grid_fname(side, interval));

    filename = stdx::format(R"(D:\Projects.github\cpp_projects\check_tracks\encounters\by_slot\by_slot_%d_%d_3months.csv)", side, interval);
    dworld.save_slot_encounters(filename);

    filename = stdx::format(R"(D:\Projects.github\cpp_projects\check_tracks\encounters\by_time\by_time_%d_%d_3months.csv)", side, interval);
    dworld.save_time_encounters(filename);
}

void save_encounters(const std::vector<std::tuple<int, int>>& params) {

    tbb::parallel_for_each(params.begin(), params.end(), [&](const std::tuple<int, int>& p) {
        int side = std::get<0>(p);
        int interval = std::get<1>(p);

        save_encounters(side, interval);
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

void save_time_encounters(std::vector<std::tuple<int, int>>& params) {

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

void save_slot_encounters(std::vector<std::tuple<int, int>>& params) {

    tbb::parallel_for_each(params.begin(), params.end(), [&](const std::tuple<int, int>& p) {
        int side = std::get<0>(p);
        int interval = std::get<1>(p);

        save_slot_encounters(side, interval);
    });
}

// --------------------------------------------------------------------------
// Simulate
// --------------------------------------------------------------------------


void simulate(const DiscreteWorld& dworld, Infections& infections,
              int i, const s_users& infected) {

    infections.infected(infected);
    infections.propagate();

    int side = dworld.side();
    int interval = dworld.interval();

    std::string dir = stdx::format(R"(D:\Projects.github\cpp_projects\check_tracks\infections\%d_%d)", side, interval);
    create_directory(path(dir));

    std::string filename = stdx::format(
        R"(%s\infections_%d_%d_%03d_3months.csv)",
        dir.c_str(), side, interval, i);

    infections.save(filename, time_duration(24, 0, 0));
}


// --------------------------------------------------------------------------

void simulate(int side, int interval, const vs_users& vinfected) {
    DiscreteWorld dworld;
    dworld.load(grid_fname(side, interval));

    Infections infections;
    infections
        .contact_range(2)
        .infection_rate(.01)
        .latent_days(5)
        .removed_days(15)
        .contact_mode(none, 1.)
        .dworld(dworld);

    for (int i : stdx::range(vinfected.size())) {
        // quota of users infected at the start of simulation
        simulate(dworld, infections, i, vinfected[i]);
    }
}

void simulate(const std::vector<std::tuple<int, int>>& params) {
    double quota = 0.05;
    int nsims = 100;

    // vector of random infected users
    vs_users vinfected;

    // generated the vector of random infected users
    {
        DiscreteWorld dworld;
        dworld.load(grid_fname(100, 60));

        for(int i : stdx::range(nsims)) {
            s_users infected = dworld.users(quota);
            vinfected.push_back(infected);
        }
    }

    tbb::parallel_for_each(params.begin(), params.end(), [&](const std::tuple<int, int>& p) {
        int side = std::get<0>(p);
        int interval = std::get<1>(p);

        simulate(side, interval, vinfected);
    });
}

void simulate_test(){
    DiscreteWorld dworld;
    dworld.load(grid_fname(100, 60));

    // vector of random infected users
    vs_users vinfected;

    s_users infected = dworld.users(0.05);
    vinfected.push_back(infected);

    simulate(100, 60, vinfected);
}

// --------------------------------------------------------------------------
// End
// --------------------------------------------------------------------------

