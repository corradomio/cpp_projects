//
// Created by Corrado Mio on 19/09/2020.
//
#include <algorithm>
#include <boostx/date_time_op.h>
#include "infections.h"
#include <stdx/cmath.h>
#include <stdx/ranges.h>
#include <stdx/to_string.h>

using namespace hls::khalifa::summer;
using namespace stdx::math;
using namespace boost::posix_time;


// --------------------------------------------------------------------------
// state_t
// --------------------------------------------------------------------------

void state_t::infective(int t) {
    _prob[t] = 1.;
}


int state_t::select(int t) const {
    int p = -1;
    for (auto it = _prob.cbegin(); it != _prob.cend(); ++it) {
        if (it->first > t)
            break;
        else
            p = it->first;
    }
    return p;
}


double state_t::prob(int t) const {
    int s = select(t);
    double p = (s == -1) ? 0. : _prob.at(s);

    if (p == 0.)
        return 0.;

    int lts = inf_p->latent_days_ts();
    int rts = inf_p->removed_days_ts();

    if (rts > 0 && (t-infected) >= rts)
        return 0.;
    if (lts > 0 && (t-infected) < lts)
        return 0.;

    return p;
}


double state_t::update(int t, double u) {
    int s = select(t);
    double p = (s == -1) ? 0. : _prob.at(s);
    p = 1. - (1. - p)*(1. - u);

    if (p == 0)
        return p;

    _prob[t] = p;

    if (infected == invalid)
        infected = t;

    return p;
}



// --------------------------------------------------------------------------
// Infections
// --------------------------------------------------------------------------

/// Quota [0,1] of infected users
Infections& Infections::infected(double quota) {
    int n = int(quota*dworld().users().size());
    return infected(n);
}


/// Number of infected users
Infections& Infections::infected(int n) {
    const s_users& susers = dworld().users();

    std::vector<user_t> users(susers.begin(), susers.end());
    int nUsers = users.size();

    std::unordered_set<user_t> selected;

    while(n != selected.size()) {
        const user_t& user = users[rnd.next_int(nUsers)];
        selected.insert(user);
    }

    return infected(selected);
}


/// Set the list of infected ids
Infections& Infections::infected(const s_users& users) {
    _infected.clear();
    _infected.insert(users.begin(), users.end());
    _infections.clear();
    return *this;
}


// --------------------------------------------------------------------------


const s_users & Infections::apply_contact_model(int t, const s_users& uset) {
    int tsday;
    switch (_cmode) {
        case contact_mode::none:
            return uset;
        case contact_mode::random:
            _cmode_users.clear();
            for(const user_t& user : uset)
                if (rnd.next_double() <= _cmode_prob)
                    _cmode_users.insert(user);
            return _cmode_users;
        case contact_mode::daily:
            tsday = t/dts;
            if (_cmode_day == tsday)
                return _cmode_users;
            _cmode_day = tsday;
            _cmode_users.clear();
            for(const user_t& user : uset)
                if (rnd.next_double() <= _cmode_prob)
                    _cmode_users.insert(user);
            return _cmode_users;
        case contact_mode::user:
            if (_cmode_day == 0)
                return _cmode_users;
            _cmode_day = 0;
            for(const user_t& user : uset)
                if (rnd.next_double() <= _cmode_prob)
                    _cmode_users.insert(user);
            return _cmode_users;
        default:
            return uset;
    }
}


Infections& Infections::propagate() {
    std::cout << "Infection::propagate ..." << std::endl;

    init_world();
    init_infected();
    propagate_infection();

    std::cout << "Infection::end" << std::endl;
    return *this;
}

/**
 * Initialize the world parameters
 */
void Infections::init_world() {
    std::cout << "Infection::init_world" << std::endl;

    // cell side
    double D = dworld().side();

    // (d/D)^2
    double dratio = sq(d/D);
    // Dt: infection's prob in a single time slot
    double DT = dworld().interval_td()/time_duration(24,0,0);
    // (1-exp(-beta*DT))
    double betadt = (1 - exp(-beta * DT));

    // factor of infection
    tau = betadt*dratio;

    // one day in time slots
    dts = int(time_duration(24, 0, 0)/dworld().interval_td());
    // l in time slots
    lts = l*dts;
    // m in time slots
    mts = m*dts;
}


void Infections::init_infected() {
    std::cout << "Infection::init_infected" << std::endl;

    std::unordered_set<user_t> processed;

    const std::map<int, vs_users> &encs = dworld().get_time_encounters();

    // initialize the '_infections' data structure
    for (const user_t& user : dworld().users())
        _infections[user].inf(this).clear();

    // scan the encounters to identify the timeslot when an infected user
    // is encountered for the first time. This timeslot MINUS the lts (latent
    // time slots) will be the 'infected' timeslot.
    bool complete = false;
    for (auto it = encs.cbegin(); it != encs.cend() && !complete; ++it) {
        int t = it->first;
        const vs_users& vsusers = it->second;

        for (auto vit = vsusers.cbegin(); vit != vsusers.end() && !complete; ++vit) {
            const s_users& users = *vit;
            for (auto uit = users.cbegin(); uit != users.cend() && !complete; ++uit) {
                user_t user = *uit;

                // it is not an infected user
                if (!stdx::contains(_infected, user))
                    continue;
                // it is an already processed user
                if (stdx::contains(processed, user))
                    continue;

                // it is:
                // - an infected user
                // - a not already processed user

                // std::unordered_map<user_t, state_t> _infections

                int inft = t - this->latent_days_ts();
                _infections[user].infective(inft);
                //_infections[inft][user].gprob = 1.;

                // register the infected user
                processed.insert(user);
                complete = (_infected.size() == processed.size());
            }
        }
    }

    // set the NOT infected users
    //for (const user_t& user : dworld().users())
    //    if (!stdx::contains(_infected, user))
    //        _infections[user].update(0, 0.)
}


void Infections::propagate_infection() {

    const std::map<int, vs_users>& encs = dworld().get_time_encounters();

    // set the parent
    for (const user_t& user : dworld().users())
        _infections[user].inf(this).clear();
     // set initial infected
    for (const user_t& user : _infected)
        _infections[user].infective(0);

    // simple counters for logging
    int n = encs.size();
    int i = 0;

    //
    //  1) for each time slot
    //      2) for each users set
    //          3) apply the contact model
    //          4) compute the aggregate infection probability
    //          5) update the infections probability for each user
    //

    // 1) for each time slot
    for(auto it = encs.cbegin(); it != encs.cend(); ++it) {
        int t = it->first;
        const vs_users& usets = it->second;

        // logging
        if ((++i)%1000 == 0)
            std::cout << "  " << i << "/"<< n << "\r";

        // 2) for each users set
        for(auto uit = usets.cbegin(); uit != usets.cend(); ++uit) {
            const s_users& uset = *uit;

            // 3) apply the contact model
            s_users users = apply_contact_model(t, uset);

            // 4) compute the aggregate infection probability
            //double prob = compute_aggregate_prob(t, users);
            //
            // 5) update the infections probability for all users
            //update_prob(t, users, prob);

            // for all users in the set
            for (const user_t& user : users) {
                // 4.0) compute the aggregate infection probability, excluding 'user'
                double aprob = compute_aggregate_prob(t, user, users);

                // 4.1) update the daily components of the user global prob
                update_daily(t, user, users);

                // 5) update the infections probability for 'user'
                update_prob(t, user, aprob);
            }
        }
    }

}


///**
// * Compute teh aggregate probability al time slot t
// *
// * @param t     time slot
// * @param users set of users to consider
// * @return aggregate probability
// */
//double Infections::compute_aggregate_prob(int t, const s_users &users) {
//    // aggregate prom
//    double ap = 1.;
//
//    for (const user_t& user : users) {
//        // user prob
//        double up = _infections[user].prob(t);
//
//        if (up != 0.)
//            ap *= (1. - tau*up);
//    }
//    return 1. - ap;
//}


///**
// * Update the internal propability
// * @param t
// * @param users
// * @param aprob
// */
//void Infections::update_prob(int t, const s_users &users, double aprob) {
//    for (const user_t& user : users) {
//        _infections[user].update(t, aprob);
//    }
//}


double Infections::compute_aggregate_prob(int t, const user_t& user, const s_users &users) {
    // aggregate prom
    double ap = 1.;

    for (const user_t& other : users) {
        if (other == user)
            continue;

        // 'other' prob
        double op = _infections[other].prob(t);
        if (op != 0.)   // DEBUG
            ap *= (1. - tau * op);
    }
    return 1. - ap;
}

void Infections::update_prob(int t, const user_t& user, double aprob) {
    _infections[user].update(t, aprob);
}

void Infections::update_daily(int t, const user_t& user, const s_users &users) {
    // daily t: last time slot of the day
    int dt = t/dts + (dts-1);

    for (const user_t& other : users) {
        if (other == user)
            continue;

        // 'other' prob
        double op = _infections[other].prob(t);

        // update user/other prob
        double uop = _daily_contacts[dt][user][other];
        _daily_contacts[dt][user][other] = 1 - (1 - uop)*(1 - tau * op);
    }
}

// --------------------------------------------------------------------------
// IO
//

//static int max(int x, int y) { return x > y ? x : y; }


void Infections::save(const std::string& filename, const time_duration& interval) const {
    std::cout << "Infections::saving in " << filename << " ..." << std::endl;

    const std::map<int, vs_users>& encs = dworld().get_time_encounters();

    save_info(filename);
    save_table(filename, interval);
    save_daily(filename);

    std::cout << "Infections::done" << std::endl;
}

static std::string with_ext(const std::string& filename, const std::string& ext) {
    int pos = filename.rfind('.');
    return filename.substr(0, pos) + ext;
}


void Infections::save_info(const std::string& filename) const {
    std::ofstream ofs(with_ext(filename, "_info.txt"));
    ofs << "# " << "\n"
        << "# side          : "<< dworld().side() << " m\n"
        << "# interval      : "<< dworld().interval() << " min (0 -> 5s)\n"
        << "# n users       : "<< dworld().users().size() << "\n"
        << "# " << "\n"
        << "# contact_range : " << contact_range() << " m\n"
        << "# infection_rate: " << infection_rate() << "/day\n"
        << "# latent_days   : " << latent_days() << " days\n"
        << "# removed_days  : " << removed_days() << " days\n"
        << "# " << "\n"
        << "# n_infected    : " << _infected.size() << "\n"
        << "# infected      : " << stdx::str(_infected) << "\n"
        << "# "
        << std::endl;
}


void Infections::save_table(const std::string& filename, const time_duration& interval) const {
    // header
    std::ofstream ofs(with_ext(filename, ".csv"));
    ofs << "\"timestamp\"";
    for (const user_t& user : dworld().users())
        ofs << ",\"" << user << "\"";
    ofs << std::endl;

    // slots per day
    int dslots = max(1, (int)(interval/dworld().interval_td()));
    // end (latest) slot
    int eslot = (ptime(date(2020, 4, 1)) - ptime(date(2020,1,1)))/dworld().interval_td();

    for(int t=0; t < eslot; t += dslots) {

        ofs << t;

        for (const user_t &user : dworld().users()) {
            double prob = _infections.at(user).prob(t);
            ofs << stdx::format(",%.5g", prob);
        }
        ofs << std::endl;
    }

}


void Infections::save_daily(const std::string& filename, bool zeros=false) const {
    std::ofstream ofs(with_ext(filename, "_daily.csv"));
    ofs << R"("timestamp","user","other","prob")" << std::endl;

    // std::map<int, std::unordered_map<user_t, std::unordered_map<user_t, double>>> _contacts;

    // t iterator
    for (auto tit = _daily_contacts.cbegin(); tit != _daily_contacts.cend(); ++tit) {
        int t = tit->first;

        // user/other map
        const std::unordered_map<user_t, std::unordered_map<user_t, double>>& uomap = tit->second;

        // user iterator
        for(auto uit = uomap.cbegin(); uit != uomap.cend(); ++uit) {
            const user_t& user = uit->first;

            // other map
            const std::unordered_map<user_t, double>& omap = uit->second;

            // other iterator
            for (auto oit = omap.begin(); oit != omap.cend(); ++oit) {
                const user_t& other = oit->first;
                double prob = oit->second;

                if (!zeros || prob > 0) {
                    ofs << t << "," << user << "," << other << "," << prob << std::endl;
                }
            }
        }
    }
}