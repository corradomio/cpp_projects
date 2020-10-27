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
#include <stdx/unordered_bag.h>
#include <stdx/random.h>
#include "dworld.h"


namespace hls {
namespace khalifa {
namespace summer {

    const int invalid = -9999;

    // ----------------------------------------------------------------------
    // ustate_t
    // ----------------------------------------------------------------------

    struct Infections;

    struct enc_t {
        user_t u1;
        user_t u2;
        double u1_after, u1_before, u2_prob;

        enc_t(){ }

        enc_t(const user_t& u1, const user_t& u2, double u1a, double u1b, double u2p)
            :u1(u1), u2(u2), u1_after(u1a), u1_before(u1b), u2_prob(u2p)
        { }

        //void set(double u1a, double u1b, double u2p) {
        //    if (count == 0) {
        //        u1_after = u1a;
        //        u1_before = u1b;
        //        u2_prob = u2p;
        //        count = 1;
        //    }
        //    else {
        //        u1_after = u1a;
        //        count += 1;
        //    }
        //}
    };

    // day -> user1 -> user2 -> {
    typedef std::map<int, std::unordered_map<user_t, std::vector<enc_t>>> daily_encs_t;
    typedef std::unordered_map<user_t, std::vector<enc_t>> users_encs_t;
    typedef std::vector<enc_t> user_encs_t;

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
        int _lts;       // latent time slots
        int _rts;       // removed time slots
        int _infected;  // time slot when received the infection
        double _prob;
    public:
        state_t(){ }

        void set(int lts, int rts);

        // initial infected user (prob = 1)
        void infective(int t);
        void not_infected(int t);

        int infected() const { return _infected; }

        // get & set & update
        double prob(int t) const;
        double prob() const { return _prob; }

        // update the probability as p' = 1 - (1-p)(1-u)
        double update(int t, double u);

        // update the probability as: p' = p*(1-r*p)
        double daily(int t, double r);
    };

    typedef std::unordered_map<user_t, state_t> user_state_t;

    class Infections {

        const DiscreteWorld* dworld_p;

        //
        // Parameters
        //

        int d;          // contact_range (in meters)
        double beta;    // infection_rate (infections/day)
        int l;          // latent_days: n of days before to became infectious.
                        // 0 -> immediately
        int m;          // removed_days: n of days after the first contact to became NOT infectious.
                        // 0 -> forever
        double t;       // test probability [0,1]
        long seed;      // random seed;

        //
        // Implementation
        //

        double dt;      // time slot in days
        double tau;     // (1-exp(-beta*delta_T))*(d/D)^2   D: side

        int lts;        // l in 'time slots'
        int rts;        // m in 'time slots'
        int dts;        // 1 day in 'time slots'

        // starting list of infected users
        s_users _infected;

        // infection status for each user
        // user -> infected
        //         infected = [state_t0, state_t1,...]
        user_state_t _infections;

        // daily infections
        // day -> u1 -> u2 -> [u1_after, u1_before, u2_prob]
        daily_encs_t _daily_infections;

        // contact modes
        //      none
        //      random
        //      user
        //      daily
        contact_mode _cmode;        // contact mode
        double       _cmode_prob;   // contact probability
        int          _cmode_day;
        s_users      _cmode_users;

        // random generator
        stdx::random_t rnd;

        // if to dump only infections
        bool _only_infections;

    public:
        Infections();

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
        /// test probability
        Infections& test_prob(double tp) { this->t = tp; return *this; }
        double      test_prob() const { return this->t; }

        /// contact mode:
        Infections& contact_mode(const contact_mode cm, double cmp) {
            this->_cmode = cm;
            this->_cmode_prob = cmp;
            return *this;
        }

        /// Quota [0,1] of infected users
        Infections& infected(double quota);
        /// Number of infected users
        Infections& infected(int n);
        /// Select the list of infected users
        Infections& infected(const s_users& users);

        /// Initial list of infected users
        const s_users& infected() const { return _infected; }

        // ------------------------------------------------------------------
        // Propagate
        // ------------------------------------------------------------------

        /// Simulate
        Infections& propagate();

        // ------------------------------------------------------------------
        // Save results
        // ------------------------------------------------------------------

        Infections& only_infections(bool enable=true) {
            _only_infections = enable;
            return *this;
        }

        enum file_format { CSV, XML };;

        ///
        void save_info(const std::string& filename) const;
        void save_table(const std::string& filename, const time_duration& interval) const;
        void save_daily(const std::string& filename, file_format format) const;

    public:

        /// latent days in time slots (available AFTER 'init()')
        int latent_days_ts()  const { return lts; }
        /// removed days in time slots (available AFTER 'init()')
        int removed_days_ts() const { return rts; }
        /// one dai in time slots (available AFTER 'init()')
        int one_day_ts() const { return dts; }

    private:

        /// Reference to dworld
        const DiscreteWorld& dworld() const { return *dworld_p; }

        /// Initialize the infected users
        void init_simulation();
        void init_infected();

        /// propagate the infection
        void propagate_infection();

        /// Apply the contact model
        const s_users& apply_contact_model(int t, const s_users& uset);

        /// Compute the aggregated user infected probabilities
        /// \param t        time slot
        /// \param users    users
        /// \return         aggregate probability
        //double compute_aggregate_prob(int t, const user_t& user, const s_users& users);

        /// Update the users infected probability
        /// \param t        time slot
        /// \param users    users
        /// \param aprob    aggregate probability
        double update_prob(int t, const user_t& user, double aprob);

        /// Latent factor
        double latent(int t0, int t) const;

        /// Update the user probability based on daily test prob
        void daily_prob(int t, const user_t& user, double tprob);

        void save_daily_csv(const std::string& filename) const;
        void save_daily_xml(const std::string& filename) const;
    };

}}}

#endif //CHECK_TRACKS_INFECTIONS_H
