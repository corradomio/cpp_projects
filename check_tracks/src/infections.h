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

    // ----------------------------------------------------------------------
    // prob_t<N>
    // ----------------------------------------------------------------------

    template <size_t N>
    struct prob_t {
        double prob[N];

        explicit prob_t(): prob_t(0,0,0) { };
        explicit prob_t(double p0): prob_t(p0,0,0) { }
        explicit prob_t(double p0, double p1): prob_t(p0,p1,0) { }
        explicit prob_t(double p0, double p1, double p2) {
            for(size_t i=3; i<N; ++i)
                prob[i] = 0.;
            prob[0] = p0;
            prob[1] = p1;
            prob[2] = p2;
        }
        prob_t(const prob_t& p) {
            //for(size_t i=0; i<N; ++i)
            //    prob[i] = p.prob[i];
            prob = p.prob;
        }

        prob_t& operator =(const prob_t& p) {
            //for(size_t i=0; i<N; ++i)
            //    prob[i] = p.prob[i];
            prob = p.prob;
            return *this;
        }

        prob_t operator *(const prob_t& p) {
            prob_t r;
            for(size_t i=0; i<N; ++i)
                r.prob[i] = prob[i]*p.prob[i];
            return r;
        }
        prob_t operator *(int s) {
            prob_t r;
            for(size_t i=0; i<N; ++i)
                r.prob[i] = 1 - std::pow(1 - prob[i], s);
            return r;
        }
        prob_t operator *(double s) {
            prob_t r;
            for(size_t i=0; i<N; ++i)
                r.prob[i] = prob[i]*s;
            return r;
        }
        prob_t operator /(double s) {
            prob_t r;
            for(size_t i=0; i<N; ++i)
                r.prob[i] = prob[i]/s;
            return r;
        }
        prob_t operator +(const prob_t& p) {
            prob_t r;
            for(size_t i=0; i<N; ++i)
                r.prob[i] = 1 - (1 - prob[i])*(1 - p.prob[i]);
            return r;
        }

        prob_t& operator *=(const prob_t& p) {
            for(size_t i=0; i<N; ++i)
                prob[i] *= p.prob[i];
            return *this;
        }
        prob_t& operator *=(double s) {
            for(size_t i=0; i<N; ++i)
                prob[i] *= s;
            return *this;
        }
        prob_t& operator *=(int s) {
            for(size_t i=0; i<N; ++i)
                prob[i] = 1 - std::pow(1 - prob[i], s);
            return *this;
        }
        prob_t& operator /=(double s) {
            for(size_t i=0; i<N; ++i)
                prob[i] /= s;
            return *this;
        }
        prob_t& operator +=(const prob_t& p) {
            for(size_t i=0; i<N; ++i)
                prob[i] = 1 - (1 - prob[i])*(1 - p.prob[i]);
            return *this;
        }

        double operator[](size_t i) const {
            return prob[i];
        }

        double& operator[](size_t i) {
            return prob[i];
        }
    };

    template<size_t N>
    prob_t<N> operator *(double s, const prob_t<N>& p) {
        prob_t<N> r;
        for(size_t i=0; i<N; ++i)
            r.prob[i] = p.prob[i]*s;
        return r;
    }
    template<size_t N>
    prob_t<N> operator *(int s, const prob_t<N>& p) {
        prob_t<N> r;
        for(size_t i=0; i<N; ++i)
            r.prob[i] = 1 - std::pow(1 - p.prob[i], s);
        return r;
    }

    template<size_t N>
    prob_t<N> operator -(double s, const prob_t<N>& p) {
        prob_t<N> r;
        for(size_t i=0; i<N; ++i)
            r.prob[i] = s - p.prob[i];
        return r;
    }

}}}


namespace hls {
namespace khalifa {
namespace summer {

    const int invalid = -9999;

    // ----------------------------------------------------------------------
    // ustate_t
    // ----------------------------------------------------------------------

    struct Infections;

    struct ustate_t {
        const Infections* p_inf;

        double _prob;       // [0]: unconscious, tested, infected
        double _life[3];    // [0]:
        int _infected;      // timestamp when infected
        int _infective;     // infected + latent_days
        int _removed;       // infected + removed_days
        int _tested;        // timestamp when tested

        ustate_t():
            _prob(0),
            _life{1.,0.,0.},
            _infected(invalid),
            _infective(invalid),
            _removed(invalid),
            _tested(invalid){ }

        ustate_t& inf(const Infections& inf) {
            p_inf = &inf;
            return *this;
        }

        ustate_t& infected() {
            _prob = 1.;
            return *this;
        }

        double prob() const { return _prob; }
        double prob(int t) {
            if (_infected == invalid)
                return _prob;
            if (t < _infective || _removed < t)
                return 0.;
            else
                return _prob;

        }
        double update(int t, double p);

        ustate_t& tested(int t, double p);

        ustate_t& infected(int t, double p);
    };

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
        std::map<int, double> _prob;

        int select(int t) const;
    public:
        state_t(): infected(invalid){ }

        state_t& inf(Infections* ptr) { inf_p = ptr; return *this; }

        // initial infected user (prob = 1)
        void infective(int t);

        // get & set & update
        double prob(int t) const;
        double update(int t, double u);
    };

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
        double p;       // positive probability [0,1]
        int r;          // result_days: n of days to wait for the result
        long seed;      // random seed;

        //
        // Implementation
        //

        double dt;      // time slot in days
        double tau;     // (1-exp(-beta*delta_T))*(d/D)^2   D: side

        int lts;        // l in 'time slots'
        int mts;        // m in 'time slots'
        int rts;        // r in 'time slots'
        int dts;        // 1 day in 'time slots'

        // starting list of infected users
        s_users _infected;

        // infection status for each user
        // user -> infected
        //         infected = [state_0, state_1,...]
        std::unordered_map<user_t, state_t> _infections;

        // contacts status for
        //      t -> user1 -> user2 -> prob
        //std::map<int, std::unordered_map<user_t, std::vector<contact_t>>>
        //    _daily_contacts;

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

    public:
        Infections() {
            d = 2;
            beta = 0.001;
            l = 0;
            m = 0;
            t = 0.01;
            p = 0.85;
            r = 2;
            seed = 123;
            _cmode_day = -1;
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
        /// test probability
        Infections& test_prob(double tp) { this->t = tp; return *this; }
        double      test_prob() const { return this->t; }
        /// positive probability. It depends on the current
        Infections& positive_prob(double pp) { this->p = pp; return *this; }
        double      positive_prob() const { return this->p; }
        /// n of days to wait for the result after the test
        Infections& result_days(int rd) { this->r = rd; return *this; }
        int         result_days() const { return this->r; }

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

        enum file_format { CSV, XML };;

        ///
        void save_info(const std::string& filename) const;
        void save_table(const std::string& filename, const time_duration& interval) const;
        void save_daily(const std::string& filename, file_format format) const;

    public:

        /// latent days in time slots (available AFTER 'init()')
        int latent_days_ts()  const { return lts; }
        /// removed days in time slots (available AFTER 'init()')
        int removed_days_ts() const { return mts; }
        /// result days in time slots (available AFTER 'init()')
        int result_days_ts() const { return rts; }

    private:

        /// Reference to dworld
        const DiscreteWorld& dworld() const { return *dworld_p; }

        // Initialize the infected users
        void init_world();
        void init_infected();
        void propagate_infection();

        /// Apply the contact model
        const s_users& apply_contact_model(int t, const s_users& uset);

        /// Compute the aggregated user infected probabilities
        /// \param t        time slot
        /// \param users    users
        /// \return         aggregate probability
        double compute_aggregate_prob(int t, const user_t& user, const s_users& users);

        /// Update the users infected probability
        /// \param t        time slot
        /// \param users    users
        /// \param aprob    aggregate probability
        void update_prob(int t, const user_t& user, double aprob);

        void save_daily_csv(const std::string& filename) const;
        void save_daily_xml(const std::string& filename) const;
    };

}}}

#endif //CHECK_TRACKS_INFECTIONS_H
