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

    /**
     * Infection state of each user.
     * 'infected' is used to mark the time slot for the first infection
     *
     * It is necessary to use a special trick for the users marked as 'infected'
     * because, at the begin, the corrected 'infected' timeslot is unknown.
     * The trick consists in to update 'infected' the FIRST time that the
     * infection probability is updated. In this case, the correct infected timeslot
     * is
     *
     *      t - latent_days_ts
     *
     * This number can be negative but this is not a problem.
     */
    class state_t {
        const Infections* inf_p;
        int infected;                   // time slot when received the infection
        bool initial;                   // it is an 'initial infected user'
        std::map<int, double> _prob;

        int select(int t) const;
    public:
        state_t(): infected(0), initial(false){ }

        state_t& inf(Infections* ptr) { inf_p = ptr; return *this; }

        // initial infected user
        state_t& infective();

        // get & set & update
        double   prob(int t) const;
        state_t& update(int t, double u);

        state_t& clear() { _prob.clear(); return *this; }
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
        std::unordered_map<user_t, state_t> _infections;

        // if to merge user sets with not empty intersection
        //bool _users_merged;

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

        /// set the world
        Infections& dworld(const DiscreteWorld& dw){ dworld_p = &dw; return *this; }

        /// contact range (in meters)
        Infections& contact_range(int cr) { this->d = cr; return *this; }
        int         contact_range() const { return this->d; }
        /// infection rate per day
        Infections& infection_rate(double ir) { this->beta = ir; return *this; }
        double      infection_rate() const { return this->beta; }
        /// n days after infection to became infective
        Infections& latent_days(int ld) { this->l = ld; return *this; }
        int         latent_days() const { return this->l; }
        /// n days after infection to became removed
        Infections& removed_days(int rd) { this->m = rd; return *this; }
        int         removed_days() const { return this->m; }
        /// if to merge sets with a not empty intersection
        //Infections& users_merged(bool m) { this->_users_merged = m; return *this; }
        //bool        users_merged() const { return this->_users_merged; }

            /// contact mode:
        Infections& contact_mode(const contact_mode cm, double cmp) {
            this->_cmode = cm;
            this->_cmode_prob = cmp;
            return *this;
        }

        /// latent days in time slots (available AFTER 'init()')
        int latent_days_ts()  const { return lts; }
        /// removed days in time slots (available AFTER 'init()')
        int removed_days_ts() const { return mts; }

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

        Infections& simulate(int n,double quota);

        ///
        //void save(const std::string& filename) const;
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
