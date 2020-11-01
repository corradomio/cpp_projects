//
// Created by Corrado Mio on 01/11/2020.
//
#include <boost/filesystem.hpp>
#include <tbb/parallel_for_each.h>
#include <stdx/params.h>
#include "dworld.h"
#include "other.h"

using namespace boost::filesystem;
using namespace boost;
using namespace hls::khalifa::summer;


void encounters(const stdx::properties& props, int side, int interval) {
    DiscreteWorld dworld;
    dworld.load(grid_fname(props.get("worlds.dir"), side, interval));

    // t -> u1 -> {u21,...}
    const tms_users & encs = dworld.get_time_encounters();

    std::string time_enc = stdx::format("encounters/by_time/time_encounters_%d_%d.csv", side, interval);
    dworld.save_time_encounters(time_enc);

    time_enc = stdx::format("encounters/by_time/time_enc_sets_%d_%d.csv", side, interval);
    dworld.save_time_encounters(time_enc, true);

    //std::string time_slots = stdx::format("encounters/by_slots/slot_encs_%d_%d.csv", side, interval);
    //dworld.save_slot_encounters(time_slots);
}


void encounters(const stdx::properties& props) {
    std::string dir = props.get("encounters.dir");
    std::string by_slots = stdx::format(R"(%s/by_slots)", dir.c_str());
    std::string by_time  = stdx::format(R"(%s/by_time)", dir.c_str());
    create_directory(path(dir));
    create_directory(path(by_slots));
    create_directory(path(by_time));

    std::vector<int> sides = props.get_ints("sides");
    std::vector<int> intervals = props.get_ints("intervals");

    std::vector<std::tuple<int, int>> params = stdx::make_params(sides, intervals);

    if (params.size() < 4) {
        std::cout << "Sequential ..." << std::endl;

        for(std::tuple<int, int> p : params) {
            int side = std::get<0>(p);
            int interval = std::get<1>(p);

            encounters(props, side, interval);
        }
    }
    else {
        std::cout << "Parallel ..." << std::endl;

        tbb::parallel_for_each(params.begin(), params.end(), [&](const std::tuple<int, int>& p) {
            int side = std::get<0>(p);
            int interval = std::get<1>(p);

            encounters(props, side, interval);
        });
    }
}

