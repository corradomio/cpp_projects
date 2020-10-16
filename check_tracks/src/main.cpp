#include <cxxopts.h>
#include "other.h"

// --------------------------------------------------------------------------
//
// --------------------------------------------------------------------------

int main(int argc, char** argv) {
    cxxopts::Options options("tracks", "Bayesian COVID analyzer");
    options.add_options()
        ("h,help", "this help")
        ("c,config", "configuration file (default 'tracks.properties')",
            cxxopts::value<std::string>()->default_value("tracks.properties"))
    ;

    auto opts = options.parse(argc, argv);

    if (opts.count("help") > 0 || opts.count("config") == 0) {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    std::string config_file = opts["config"].as<std::string>();
    stdx::properties props(config_file);

    simulate(props);

    return 0;
}
