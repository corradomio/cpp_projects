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

    // one years back, in seconds: 1*365*24*60*60
    const int invalid = -31536000;

    // ----------------------------------------------------------------------
    // enc_t
    // ----------------------------------------------------------------------

    struct Infections;

    // ----------------------------------------------------------------------
    // enc_t
    // ----------------------------------------------------------------------

    struct enc_t {
        user_t u1;
        user_t u2;
        double u1_after, u1_before, u2_prob;

        enc_t(){ }

        enc_t(const user_t& u1, const user_t& u2, double u1a, double u1b, double u2p)
            :u1(u1), u2(u2), u1_after(u1a), u1_before(u1b), u2_prob(u2p)
        { }
    };

    //              [(u2, u1a, u1b, u2p), ...]
    typedef std::vector<enc_t> uencs_t;
    //        u1 -> [(u2, u1a, u1b, u2p), ...]
    typedef std::unordered_map<user_t, uencs_t> users_encs_t;
    // day -> u1 -> [(u2, u1a, u1b, u2p), ...]
    typedef std::map<int, users_encs_t> daily_encs_t;

    // ----------------------------------------------------------------------

    struct history_t;

    /**
     * Information of  the disease
     */
    struct disease_t {
        int latent;         // number of time slots for an agent infected but not infectious
        int asymptomatic;   // number of time slots of asymptomatic disease
        int removed;        // number of time slots of the disease life;

        double symptomatic(const history_t& histo, int t) const;
        double infective(const history_t& histo, int t) const;
        double duration(const history_t& histo, int t) const;

        bool susceptible(const history_t& histo, int t) const;
    };

    // ----------------------------------------------------------------------

    /**
     * time probability pair:  t -> prob
     */
    struct tprob_t {
        int t;
        double prob;
        tprob_t(int t, double p): t(t), prob(p) { }
    };

    /**
     * probability history: [t1->prob, ...]
     */
    struct prob_hist_t {
        std::vector<tprob_t> hist;
        //int positive;

        prob_hist_t() {
            //positive = invalid;
            hist.emplace_back(invalid, 0.);
        }

        /// time slot when it got infected
        //int when_positive() const {
        //    return positive;
        //}

        /// user infectious probability
        double get(int t) const;

        /// set the infection probability
        void set(int t, double p);

        /// p' = 1 - (1 - p)*(1 - u)
        double update(int t, double u);

        /// p' = p*(1 - p*u)
        double scale(int t, double u);
    };

    /**
     * user infected history
     */
    class history_t {
        user_t user;                    // user
        prob_hist_t tested;             // probability to be tested
        prob_hist_t infected;           // probability to be infected
        int positive;

    public:
        /// initialize the data structure
        void set(const user_t& u, const disease_t& d, double test_prob) {
            user = u;
            tested.set(invalid, test_prob);
            positive = invalid;
        }

        /// initially not infected
        void not_infected(int t) {
            infected.set(t, 0.);
        }

        /// initially infected & infective
        void is_infected(int t) {
            infected.set(t, 1.);
            positive = t;
        }

        /// when become infected (can be 'invalid')
        int when_infected() const {
            return positive;
        }

        /// user infected probability
        double prob(int t) const {
            return infected.get(t);
        }

        void infected_update(int t, double p) {
            p = infected.update(t, p);
            if (p != 0 && positive == invalid)
                positive = t;
        }

        void infected_scale(int t, double p) {
            infected.scale(t, p);
        }

        /// user retired
        void removed(int t, bool susceptible) {
            if (!susceptible)
                infected.set(t,0);
        }

        double tested_get(int t) const {
            return tested.get(t);
        }

        double tested_update(int t, double p) {
            return tested.update(t, p);
        }
    };

    /// users history map:  user -> history
    typedef std::unordered_map<user_t, history_t> user_hist_t;

    // ----------------------------------------------------------------------

    class Infections {

        const DiscreteWorld* dworld_p;

        //
        // Parameters
        //

        int d;          // contact_range (in meters)
        double beta;    // infection_rate (infections/day)

        int a;          // asymptomaitic_days: n of days before to have symptoms
        int l;          // latent_days: n of days before to became infectious.
        int r;          // removed_days: n of days after the first contact to became NOT infectious.

        double sp;      // symptoms probability
        double tp;      // disease test probability
        double ce;      // contact efficiency

        // starting list of infected users
        s_users _infected;

        //
        // Implementation
        //

        disease_t _disease;     // disease information

        int dts;        // days in time slots
        int last_ts;    // last tieslot processed
        double tau;     // dworld infection probability

        // infection status for each user
        // user -> history
        user_hist_t _infections;

        // daily infections
        // day -> u1 -> u2 -> [u1_after, u1_before, u2_prob]
        daily_encs_t _daily_infections;

        // random generator
        stdx::random_t rnd;
        long seed;      // random seed;

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
        double      infection_rate() const    { return this->beta; }

        /// n days after infection asymptomatic
        Infections& asymptomatic_days(int ad) { this->a = ad; return *this; }
        int         asymptomatic_days() const { return this->a; }
        /// n days after infection to became infective
        Infections& latent_days(int ld) { this->l = ld; return *this; }
        int         latent_days() const { return this->l; }
        /// n days after infection to became removed
        Infections& removed_days(int rd) { this->r = rd; return *this; }
        int         removed_days() const { return this->r; }

        /// test probability
        Infections& test_prob(double tp) { this->tp = tp; return *this; }
        double      test_prob() const    { return this->tp; }
        /// symptomatic probability
        Infections& symptomatic_prob(double sp) { this->sp = sp; return *this; }
        double      symptomatic_prob() const    { return this->sp; }

        /// contact mode:
        Infections& contact_efficiency(double ce) { this->ce = ce; return *this; }
        double      contact_efficiency() const    { return this->ce; }

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

        /// saved file formats
        enum file_format { CSV, XML };;

        /// save simulation infos
        void save_info(const std::string& filename) const;
        /// save infection status day per day
        void save_table(const std::string& filename, const time_duration& interval) const;

        /// save the encounters & infection probability, day per day
        void save_daily(const std::string& filename, file_format format) const;

    public:

        /// disease information (available AFTER 'init()')
        const disease_t& disease() const { return _disease; }

    private:

        /// reference to dworld
        const DiscreteWorld& dworld() const { return *dworld_p; }

        /// initialize the infected users
        void init_simulation();
        void init_infected();
        void init_propagation();

        /// propagate the infection
        void propagate_infection();

        /// update the probability of u1 after an encounter with u2
        void update_for_encounter(int t, const user_t& u1, const user_t& u2);
        /// update the probability of u1 because theday is changed
        void update_for_newday(int t, const user_t& u1);

        // IO
        void save_daily_csv(const std::string& filename) const;
        void save_daily_xml(const std::string& filename) const;
    };

}}}

#endif //CHECK_TRACKS_INFECTIONS_H
