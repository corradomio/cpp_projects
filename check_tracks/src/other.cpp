//
// Created by Corrado Mio on 19/09/2020.
//

#include <csvstream.h>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <tbb/parallel_for_each.h>
#include "infections.h"
#include <stdx/ranges.h>
#include <stdx/properties.h>
#include <stdx/to_string.h>

using namespace boost::filesystem;
using namespace boost;
using namespace hls::khalifa::summer;



// --------------------------------------------------------------------------
// Parameters
// --------------------------------------------------------------------------

std::string grid_fname(const std::string& worlds_dir, int side, int interval) {
    std::string fname = stdx::format(
    R"(%s\tracksgrid_%d_%d_3months.bin)",
        worlds_dir.c_str(),
        side, interval);
    return fname;
}

std::vector<std::tuple<int, int>> make_params(const stdx::properties& props) {
    std::vector<std::tuple<int, int>> params;

    std::vector<int> sides = props.get_ints("sides");
    std::vector<int> intervals = props.get_ints("intervals");

    for (int side : sides)
        for (int interval: intervals)
            params.emplace_back(side, interval);

    return params;
}

// --------------------------------------------------------------------------
// Simulate
// --------------------------------------------------------------------------

void simulate(const stdx::properties& props,
              const DiscreteWorld& dworld, Infections& infections,
              const s_users& infected, int i) {

    // initialize the infected users
    infections.infected(infected);

    // propagate the infections
    infections.propagate();
}


void save_results(const stdx::properties& props,
                  const DiscreteWorld& dworld,
                  const Infections& infections,
                  int i) {

    int side = dworld.side();
    int interval = dworld.interval();

    std::string dir = stdx::format(R"(%s\%d_%d)",
                                   props.get("infections.dir").c_str(),
                                   side, interval);
    create_directory(path(dir));

    std::string filename = stdx::format(
        R"(%s\infections_%d_%d_%03d_3months.csv)",
        dir.c_str(), side, interval, i);

    infections.save_info(filename);
    infections.save_table(filename, time_duration(24, 0, 0));
    infections.save_daily(filename, Infections::file_format::XML);
}


// --------------------------------------------------------------------------

void simulate(const stdx::properties& props, int side, int interval, const vs_users& vinfected) {
    std::string data_dir = props.get("worlds.dir");
    DiscreteWorld dworld;
    dworld.load(grid_fname(data_dir, side, interval));

    Infections infections;
    infections
        .contact_range(props.get("contact_range", 2))
        .infection_rate(props.get("infection_rate", .01))
        .latent_days(props.get("latent_days", 5))
        .removed_days(props.get("removed_days", 15))
        .test_prob(props.get("test_prob", 0.01))
        .only_infections(props.get("only_infections", true))
        .contact_mode(
            static_cast<contact_mode>(props.get("contact_mode", {"none", "random", "daily", "user"})),
            props.get("contact_prob", 1.0))
        .dworld(dworld);

    for (int i : stdx::range(vinfected.size())) {
        simulate(props, dworld, infections, vinfected[i], i);
        save_results(props, dworld, infections, i);
    }
}

void simulate(const stdx::properties& props) {

    double quota = props.get("quota", 0.05);
    int nsims = props.get("nsims", 1);

    std::vector<std::tuple<int, int>> params = make_params(props);

    // vector of random infected users
    vs_users vinfected;

    // generated the vector of random infected users
    DiscreteWorld dworld;
    dworld.load(grid_fname(props.get("worlds.dir"), 100, 60));
    {
        //DiscreteWorld dworld;
        //dworld.load(grid_fname(props.get("worlds.dir"), 100, 60));

        for(int i : stdx::range(nsims)) {
            s_users infected = dworld.users(quota);

            std::cout << (i+1) << " infected: "<< stdx::str(infected) << std::endl;

            vinfected.push_back(infected);
        }
    }

    if (params.size() < 4) {
        std::cout << "Sequential simulations ..." << std::endl;

        for(std::tuple<int, int> p : params) {
            int side = std::get<0>(p);
            int interval = std::get<1>(p);

            simulate(props, side, interval, vinfected);
        }
    }
    else {
        std::cout << "Parallel simulations ..." << std::endl;

        tbb::parallel_for_each(params.begin(), params.end(), [&](const std::tuple<int, int>& p) {
            int side = std::get<0>(p);
            int interval = std::get<1>(p);

            simulate(props, side, interval, vinfected);
        });
    }
}


// --------------------------------------------------------------------------
// world
// --------------------------------------------------------------------------

void world(const stdx::properties& props, int side, int interval) {
    std::string data_dir = props.get("worlds.dir");
    std::string dir = props.get("encounters.dir");
    std::string by_slots = stdx::format(R"(%s/by_slots)", dir.c_str());
    std::string by_time  = stdx::format(R"(%s/by_time)", dir.c_str());

    DiscreteWorld dworld;
    dworld.load(grid_fname(data_dir, side, interval));

    std::string slot_encs = stdx::format(R"(%s/by_slot_%d_%d_3months.csv)",
                                         by_slots.c_str(), side, interval);
    dworld.save_slot_encounters(slot_encs);

    std::string time_encs = stdx::format(R"(%s/by_time_%d_%d_3months.csv)",
                                         by_time.c_str(), side, interval);
    dworld.save_time_encounters(time_encs);
}

void world(const stdx::properties& props) {
    std::string dir = props.get("encounters.dir");
    std::string by_slots = stdx::format(R"(%s/by_slots)", dir.c_str());
    std::string by_time  = stdx::format(R"(%s/by_time)", dir.c_str());
    create_directory(path(dir));
    create_directory(path(by_slots));
    create_directory(path(by_time));

    std::vector<std::tuple<int, int>> params = make_params(props);

    if (params.size() < 4) {
        std::cout << "Sequential ..." << std::endl;

        for(std::tuple<int, int> p : params) {
            int side = std::get<0>(p);
            int interval = std::get<1>(p);

            world(props, side, interval);
        }
    }
    else {
        std::cout << "Parallel ..." << std::endl;

        tbb::parallel_for_each(params.begin(), params.end(), [&](const std::tuple<int, int>& p) {
            int side = std::get<0>(p);
            int interval = std::get<1>(p);

            world(props, side, interval);
        });
    }
}

// --------------------------------------------------------------------------
// End
// --------------------------------------------------------------------------

