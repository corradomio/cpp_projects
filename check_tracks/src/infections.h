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

    class state_t {
        const DiscreteWorld* dworld_p;
        //int susceptible;    // time slot not infected
        //int exposed;        // time slot infected but not infective
        //int infective;      // time slot when infective
        //int removed;        // time slot when removed from the infective pool
        std::map<int, double> _prob;

        int select(int t) const;
    public:
        state_t& dworld(const DiscreteWorld& dworld_) { dworld_p = &dworld_; return *this; }

        state_t& prob(int t, double p);
        double   prob(int t) const;
    };

    class Infections {

        const DiscreteWorld* dworld_p;

        //
        // Parameters
        //

        int d;          // contact range (in meters)
        double beta;    // rate of infection (infections/day)
        int l;          // latent days before to became infectious.
                        // 0 -> immediately
        int m;          // n of days after the first contact to became NOT infectious.
                        // 0 -> forever
        long seed;      // random seed;

        //
        // Implementation
        //

        double dratio;          // (d/D)^2   D: side
        double betadt;          // (1-exp(-beta*delta_T))

        int lts;                // l in 'time slots'
        int mts;                // m in 'time slots'
        int dts;                // 1 day in 'time slots'

        contact_mode _cmode;    // contact mode
        double _cmode_prob;     // contact probability

        // starting list of infected users
        s_users _infected;

        // infection status for each user
        // user -> infected
        //         infected = [state_0, state_1,...]
        //std::unordered_map<std::string, std::vector<state_t>> infections;
        std::unordered_map<user_t, state_t> _infections;

        // random generator
        stdx::random_t rnd;

    public:
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

        /// Set the world
        Infections& dworld(const DiscreteWorld& dworld_){ dworld_p = &dworld_; return *this; }

        Infections& contact_range(int d_) { this->d = d_; return *this; }
        Infections& infection_rate(double beta_) { this->beta = beta_; return *this; }
        Infections& latent_days(int l_) { this->l = l_; return *this; }
        Infections& removed_days(int m_) { this->m = m_; return *this; }
        Infections& contact_mode(const contact_mode cmode, const double cmode_prob) {
            this->_cmode = cmode;
            this->_cmode_prob = cmode_prob;
            return *this;
        }

        /// Quota [0,1] of infected users
        Infections& infected(double quota);
        /// Number of infected users
        Infections& infected(int n);
        /// Select the list of infected users
        Infections& infected(const s_users& users);

        /// Initialize the simulator
        Infections& init();
        /// Simulate
        Infections& propagate();

        ///
        void save(const std::string& filename) const;
        void save(const std::string& filename, const time_duration& interval) const;

    private:
        /// Reference to dworld
        const DiscreteWorld& dworld() const { return *dworld_p; }

        /// Compute the aggregated user infected probabilities
        /// \param t        time slot
        /// \param users    users
        /// \return         aggregate probability
        double compute_aggregate_prob(int t, const s_users& users);

        /// Update the users infected probability
        /// \param t        time slot
        /// \param users    users
        /// \param aprob    aggregate probability
        void   update_prob(int t, const s_users& users, double aprob);
    };

}}}

#endif //CHECK_TRACKS_INFECTIONS_H
