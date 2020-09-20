//
// Created by Corrado Mio on 19/09/2020.
//

#ifndef CHECK_TRACKS_INFECTIONS_H
#define CHECK_TRACKS_INFECTIONS_H

#include <string>
#include <vector>
#include <map>
#include <list>
#include <unordered_map>
#include <stdx/ref_unordered_map.h>
#include <stdx/random.h>
#include "dworld.h"

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

        state_t() { }
        state_t(int t, double p):t(t), prob(p){ }
        state_t(const state_t& s): t(s.t),prob(s.prob){ }

        state_t& operator =(const state_t& s) {
            t = s.t;
            prob = s.prob;
            return *this;
        }
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
        contact_mode cmode;     // contact mode
        double cmode_prob;     // contact probability

        DiscreteWorld const* dworld_p;

        // starting list of infected ids
        std::unordered_set<std::string>  infected;

        // infection status for each id
        // id -> infected
        std::unordered_map<std::string, std::vector<state_t>> infections;

        // random generator
        stdx::random_t rnd;

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

        // ------------------------------------------------------------------
        // Properties
        // ------------------------------------------------------------------

        Infections& set_d(int d_) { this->d = d_; return *this; }
        Infections& set_beta(double beta_) { this->beta = beta_; return *this; }
        Infections& set_l(int l_) { this->l = l_; return *this; }
        Infections& set_m(int m_) { this->m = m_; return *this; }
        Infections& set_contact_mode(contact_mode cmode, double cmode_prob) {
            this->cmode = cmode;
            this->cmode_prob = cmode_prob;
            return *this;
        }

        Infections& set_dworld(const DiscreteWorld& dworld_);

        /// Quota [0,1] of infected ids
        Infections& set_infected(float quota){ return set_infected(double(quota)); }
        /// Quota [0,1] of infected ids
        Infections& set_infected(double quota);
        /// Number of infected ids
        Infections& set_infected(int n);
        /// Select the list of infected ids
        Infections& set_infected(const std::unordered_set<std::string>& ids);

        void propagate();

    private:
        const DiscreteWorld& dworld() { return *dworld_p; }
        ref::unordered_map<int, std::vector<std::unordered_set<std::string>>>
            get_all_encounters();

    };

}}}

#endif //CHECK_TRACKS_INFECTIONS_H
