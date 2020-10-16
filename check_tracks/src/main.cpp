#include <cxxopts.h>
#include "other.h"
#include <stdx/properties.h>


// --------------------------------------------------------------------------
//
//std::vector<std::tuple<int, int>> make_params(const stdx::properties& props) {
//    std::vector<std::tuple<int, int>> params;
//
//    //std::vector<int> sides{5,10,20,50,100};
//    //std::vector<int> intervals{1,5,10,15,30,60};
//
//    std::vector<int> sides = props.get_ints("sides");
//    std::vector<int> intervals = props.get_ints("intervals");
//
//    for (int side : sides)
//        for (int interval: intervals)
//            params.emplace_back(side, interval);
//
//    return params;
//}

// --------------------------------------------------------------------------

int main(int argc, char** argv) {
    cxxopts::Options options("tracks", "Bayesian COVID analyzer");
    options.add_options()
        ("h,help", "this help")
        ("c,config", "configuration file", cxxopts::value<std::string>()->default_value("tracks.properties"))
    ;

    auto opts = options.parse(argc, argv);

    if (opts.count("help") > 0 || opts.count("config") == 0) {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    std::string config_file = opts["config"].as<std::string>();
    stdx::properties props(config_file);

    //std::vector<std::tuple<int, int>> params = make_params(props);
    //
    //save_encounters(params);
    //save_slot_encounters();
    //save_time_encounters();

    simulate(props);

    return 0;
}
