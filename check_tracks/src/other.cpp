//
// Created by Corrado Mio on 19/09/2020.
//

#include <csvstream.h>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <tbb/parallel_for_each.h>
#include <stdx/ranges.h>
#include <stdx/properties.h>
#include <stdx/to_string.h>
#include <stdx/strings.h>
#include <stdx/params.h>
#include "infections.h"
#include "other.h"

using namespace boost::filesystem;
using namespace boost;
using namespace hls::khalifa::summer;



// --------------------------------------------------------------------------
// Static
// --------------------------------------------------------------------------

//DiscreteWorld dworld;
//int w_side = 0;
//int w_interval = 0;

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

// --------------------------------------------------------------------------
// infeced_users
// --------------------------------------------------------------------------

void infected_users(const stdx::properties& props) {
    // dworld used to retrieve the list of users
    DiscreteWorld dworld;
    dworld.load(grid_fname(props.get("worlds.dir"), 100, 60));

    // some parameters
    int nsims = props.get("nsims", 1);

    double quota = props.get("quota", 0.0);
    int n_infected = props.get("n_infected", 0);

    vs_users vinfected;
    s_users infected;

    if (quota != 0) {
        for(int i : stdx::range(nsims)) {
            infected = dworld.users(quota);
            std::cout << (i+1) << " infected[" << infected.size() << "]: "<< stdx::str(infected) << std::endl;
            vinfected.push_back(infected);
        }
    }
    else{
        for(int i : stdx::range(nsims)) {
            infected = dworld.users(n_infected);
            std::cout << (i+1) << " infected[" << infected.size() << "]: "<< stdx::str(infected) << std::endl;
            vinfected.push_back(infected);
        }
    }

    std::string infected_file = props.get("infected.file");

    save_infected(infected_file, vinfected);
}


void save_infected(const std::string file, const std::vector<std::unordered_set<int>>&  vinfected) {
    std::ofstream os;
    os.open(file.c_str());

    for (auto users : vinfected) {
        int i=0;
        for (auto user : users) {
            if (i++ > 0) os << ",";
            os << user;
        }
        os << std::endl;
    }
}

void load_infected(const std::string file, std::vector<std::unordered_set<int>>&  vinfected) {
    std::ifstream is;
    is.open(file.c_str());

    std::string line;
    std::unordered_set<int> users;

    while (getline(is, line)) {
        users.clear();

        if(line.empty())
            continue;

        std::vector<std::string> parts = stdx::split(line, ",");
        for (auto part : parts) {
            int user = atol(part.c_str());
            users.insert(user);
        }

        vinfected.push_back(users);
    }
}


// --------------------------------------------------------------------------
// Simulate
// --------------------------------------------------------------------------

void simulate(const stdx::properties& props,
              const DiscreteWorld& dworld, Infections& infections,
              const s_users& infected, int i) {

    std::cout << "Infections::simulate [" << infections.contact_efficiency() << "] " << i << std::endl;

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
    int efficiency = (int)lround(infections.contact_efficiency()*10);

    std::string dir = stdx::format(R"(%s\%d_%d_%d)",
                                   props.get("infections.dir").c_str(),
                                   side, interval, efficiency);
    create_directory(path(dir));

    std::string filename = stdx::format(
        R"(%s\infections_%02d_%02d_%02d_%03d.csv)",
        dir.c_str(), side, interval, efficiency, i);

    infections.save_info(filename);
    infections.save_table(filename, time_duration(24, 0, 0));
    //infections.save_daily(filename, Infections::file_format::XML);
}


// --------------------------------------------------------------------------

void simulate(const stdx::properties& props,
              int side,
              int interval,
              double efficiency,
              const vs_users& vinfected)
{
    std::string data_dir = props.get("worlds.dir");
    DiscreteWorld dworld;
    dworld.load(grid_fname(data_dir, side, interval));

    //if (w_side != side || w_interval != interval) {
    //    dworld.load(grid_fname(data_dir, side, interval));
    //    w_side = side;
    //    w_interval = interval;
    //}

    int nsims = props.get("nsims", 1);

    Infections infections;
    infections
        .contact_range(props.get("contact_range", 2))
        .infection_rate(props.get("infection_rate", .01))
        .latent_days(props.get("latent_days", 5))
        .removed_days(props.get("removed_days", 15))
        .asymptomatic_days(props.get("asymptomatic_days", 2))
        .test_prob(props.get("test_prob", 0.01))
        .symptomatic_prob(props.get("symptomatic_prob", 0.20))
        .contact_efficiency(efficiency)
        //.only_infections(props.get("only_infections", true))
        .dworld(dworld);

    for (int i : stdx::range(nsims)) {
        simulate(props, dworld, infections, vinfected[i], i);
        save_results(props, dworld, infections, i);
    }
}

void simulate(const stdx::properties& props) {

    bool seq = props.get("sequential", false);


    std::vector<int> sides = props.get_ints("sides");
    std::vector<int> intervals = props.get_ints("intervals");
    std::vector<double> efficiencies = props.get_doubles("contact_efficiency");
    std::vector<std::tuple<int, int, double>> params = stdx::make_params(sides, intervals, efficiencies);

    // vector of random infected users
    vs_users vinfected;
    load_infected(props.get("infected.file"), vinfected);

    if (params.size() < 4 || seq) {
        std::cout << "Sequential::simulations ..." << std::endl;

        for(std::tuple<int, int, double> p : params) {
            int side = std::get<0>(p);
            int interval = std::get<1>(p);
            double efficiency = std::get<2>(p);

            simulate(props, side, interval, efficiency, vinfected);
        }
        std::cout << "Sequential::done" << std::endl;
    }
    else {
        std::cout << "Parallel::simulations ..." << std::endl;

        tbb::parallel_for_each(params.begin(), params.end(), [&](const std::tuple<int, int, double>& p) {
            int side = std::get<0>(p);
            int interval = std::get<1>(p);
            double efficiency = std::get<2>(p);

            simulate(props, side, interval, efficiency, vinfected);
        });
        std::cout << "Parallel::done" << std::endl;
    }
}


// --------------------------------------------------------------------------
// world
// --------------------------------------------------------------------------



void world(const stdx::properties& props, int side, int interval) {

    DiscreteWorld dworld;
    std::string data_dir = props.get("worlds.dir");
    std::string dir = props.get("encounters.dir");
    std::string by_slots = stdx::format(R"(%s/by_slots)", dir.c_str());
    std::string by_time  = stdx::format(R"(%s/by_time)",  dir.c_str());

    dworld.load(grid_fname(data_dir, side, interval));

    std::string slot_users = stdx::format(R"(%s/by_slot_%d_%d.csv)", by_slots.c_str(), side, interval);
    dworld.save_slot_users(slot_users);

    std::string time_encs = stdx::format(R"(%s/by_time_%d_%d.csv)", by_time.c_str(), side, interval);
    dworld.save_time_encounters(time_encs);
}

void world(const stdx::properties& props) {
    std::string dir = props.get("encounters.dir");
    std::string by_slots = stdx::format(R"(%s/by_slots)", dir.c_str());
    std::string by_time  = stdx::format(R"(%s/by_time)", dir.c_str());
    create_directory(path(dir));
    create_directory(path(by_slots));
    create_directory(path(by_time));

    bool seq = props.get("sequential", false);

    std::vector<int> sides = props.get_ints("sides");
    std::vector<int> intervals = props.get_ints("intervals");

    std::vector<std::tuple<int, int>> params = stdx::make_params(sides, intervals);

    if (params.size() < 4 || seq) {
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

