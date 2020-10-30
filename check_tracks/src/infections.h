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

    //              [(u2, u1a, u1b, u2p), ...]
    typedef std::vector<enc_t> uencs_t;
    //        u1 -> [(u2, u1a, u1b, u2p), ...]
    typedef std::unordered_map<user_t, uencs_t> users_encs_t;
    // day -> u1 -> [(u2, u1a, u1b, u2p), ...]
    typedef std::map<int, users_encs_t> daily_encs_t;


    enum contact_mode {
        none, random, daily, user
    };

    // ----------------------------------------------------------------------

    struct bluetooth_t {
        double efficency;     // communication efficiency
    };

    // ----------------------------------------------------------------------

    /**
     * Information of  the disease
     */
    struct disease_t {
        double dt;          // delta_T
        double tau;         // (1-exp(-beta*delta_T))*(d/D)^2   D: side
        int latent;         // number of time slots for an agent infected but not infectious
        int asymptomatic;   // number of time slots of asymptomatic disease
        int removed;        // number of time slots of the disease life;

        double infective(int t0, int t);
        double symptomatic(int t0, int t);
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
        int infected;

        prob_hist_t() {
            infected = invalid;
            hist.emplace_back(invalid, 0.);
        }

        /// time slot when it got infected
        int when_infected() const {
            return infected;
        }

        /// user infection probability
        double get() const {
            size_t n = hist.size();
            return hist[n-1].prob;
        }

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
    struct history_t {
        user_t user;                    // user
        const disease_t* disease_p;     // disease information
        prob_hist_t tested;             // probability to be tested
        prob_hist_t infected;           // probability to be infected

        /// initialize the data structure
        void set(const user_t& u, const disease_t& d, double test_prob) {
            user = u;
            disease_p = &d;
            tested.set(0, test_prob);
        }

        /// initially not infected
        void not_infected(int t) {
            infected.set(t, 0.);
        }

        /// initially infected & infective
        void infective(int t) {
            infected.set(t - disease_p->latent, 1.);
        }

        /// user infected probability
        double prob() const {
            return infected.get();
        }

        /// user infectious probability
        double prob(int t) const;

        /// disease symptoms
        double symptomatic(int t) const;
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

        int l;          // latent_days: n of days before to became infectious.
        int r;          // removed_days: n of days after the first contact to became NOT infectious.
        int s;          // symptoms_days: n of days before to have symptoms

        double sp;      // symptoms probability
        double tp;      // disease test probability
        long seed;      // random seed;

        //
        // Implementation
        //

        disease_t _disease;  // disease information

        int dts;        // days in time slots

        // starting list of infected users
        s_users _infected;

        // infection status for each user
        // user -> history
        user_hist_t _infections;

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
        Infections& removed_days(int rd) { this->r = rd; return *this; }
        int         removed_days() const { return this->r; }
        /// n days after infection to have symptoms
        Infections& symptoms_days(int sd) { this->s = sd; return *this; }
        int         symptoms_days() const { return this->s; }

        /// test probability
        Infections& test_prob(double tp) { this->tp = tp; return *this; }
        double      test_prob() const { return this->tp; }
        /// symptomatic probability
        Infections& symptomatic_prob(double sp) { this->sp = sp; return *this; }
        double      symptomatic_prob() const { return this->sp; }

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

        /// saved file formats
        enum file_format { CSV, XML };;

        /// save simulation infos
        void save_info(const std::string& filename) const;
        /// save infection status day per day
        void save_table(const std::string& filename, const time_duration& interval) const;

        /// if to save only the infections
        Infections& only_infections(bool enable=true) {
            _only_infections = enable;
            return *this;
        }

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

        /// propagate the infection
        void propagate_infection();

        /// apply the contact model
        const s_users& apply_contact_model(int t, const s_users& uset);

        void update_for_encounter(int t, const user_t& u1, const user_t& u2);
        void update_for_newday(int t, const user_t& u1);

        /// Update the users infected probability
        /// \param t        time slot
        /// \param users    users
        /// \param aprob    aggregate probability
        double update_prob(int t, const user_t& user, double aprob);

        /// Update the user probability based on daily test prob
        void encounter_prob(int t, const user_t& u1, const user_t& u2);

        /// Update the user probability based on daily test prob
        void daily_prob(int t, const user_t& user);

        /// Latent factor
        double latent(int t0, int t) const;

        void save_daily_csv(const std::string& filename) const;
        void save_daily_xml(const std::string& filename) const;
    };

}}}

#endif //CHECK_TRACKS_INFECTIONS_H
