//
// Created by Corrado Mio on 19/09/2020.
//

#ifndef CHECK_TRACKS_INFECTIONS_H
#define CHECK_TRACKS_INFECTIONS_H

#include <string>
#include <vector>
#include <map>
#include <unordered_map>

namespace hls {
namespace khalifa {
namespace summer {

    struct Infections;

    enum contact_mode {
        none, random, daily, user
    };

    struct state_t {
        int t;          // time id
        double prob;    // proability of infection
    };

    struct infected_t {
        Infections* parent;
        std::string id;
        std::vector<state_t> infected;
    };

    struct Infections {

        //
        // Parameters
        //

        int d;          // contact range (in meters)
        double beta;    // rate of infection (infections/day)
        int l;          // latent days before to became infectious.
                        // 0 -> immediately
        int m;          // n of days after the first contact to became NOT infectious.
                        // 0 or -1 -> forever
        long seed;      // random seed;

        //
        // Imlementation
        //
        double dratio;          // (d/D)^2   D: side
        double betadt;          // (1-exp(-beta*delta_T))
        int start;              // l in 'time slots'
        int reset;              // m in 'time slots'
        int oneday;             // 1 day in 'time slots'
        contact_mode mode;      // contact mode
        double contact_prob;    // contact probability

        // starting list of infected ids
        std::vector<std::string>  infected_ids;

        // infection status for each id
        // id -> infected
        std::unordered_map<std::string, infected_t> infections;

        // ------------------------------------------------------------------
        //
        // ------------------------------------------------------------------

        Infections() {
            d = 2;
            beta = 0.001;
            l = 0;
            m = 0;
            seed = 123;
        }

    };

}}}

#endif //CHECK_TRACKS_INFECTIONS_H
