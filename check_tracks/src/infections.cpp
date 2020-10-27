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

void state_t::set(int lts, int rts) {
    _infected = invalid;
    _prob = 0;
    _lts = lts;
    _rts = rts;
}


void state_t::infective(int t) {
    _infected = t - _lts;
    _prob = 1.;
}


void state_t::not_infected(int t) {
    _infected = invalid;
    _prob = 0.;
}


double state_t::prob(int t) const {
    // not infected
    if (_infected == invalid)
        return 0.;
    // latent period
    if (t < _infected + _lts)
        return 0.;
    // infective period
    if (t <= _infected + _rts)
        return _prob;
    // removed
    else
        return 0.;
}


double state_t::update(int t, double u) {
    if(u != 0.)
        _prob = 1. - (1. - _prob)*(1. - u);

    if (_prob != 0 && _infected == invalid)
        _infected = t;
    return _prob;
}


double state_t::daily(int t, double r) {
    if (_prob != 0.)
        _prob = _prob*(1 - r*_prob);
    return _prob;
}


// --------------------------------------------------------------------------
// Infections
// --------------------------------------------------------------------------

Infections::Infections() {
    d = 2;
    beta = 0.001;
    l = 0;
    m = 0;
    t = 0.01;
    seed = 123;
    _cmode_day = none;
    _only_infections = true;
}

/// Quota [0,1] of infected users
Infections& Infections::infected(double quota) {
    int n = int(quota*dworld().users().size());
    return infected(n);
}


/// Number of infected users
Infections& Infections::infected(int n) {
    const s_users& susers = dworld().users();
    int nUsers = susers.size();

    // all users are infected
    if (n >= nUsers)
        return infected(susers);

    // set[use] -> array[user]
    std::vector<user_t> users(susers.begin(), susers.end());

    // selected infected users
    std::unordered_set<user_t> selected;

    // loop until |selected| == n
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
// Propagate infection
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

    init_simulation();
    init_infected();

    propagate_infection();

    std::cout << "Infection::end" << std::endl;
    return *this;
}

/**
 * Initialize the simulation parameters
 */
void Infections::init_simulation() {
    std::cout << "Infection::init_simulation" << std::endl;

    // cell side
    double D = dworld().side();

    // dt: infection's prob in a single time slot
    dt = dworld().interval_td()/time_duration(24,0,0);

    // (d/D)^2
    double dratio = sq(d/D);
    // (1-exp(-beta*dt))
    double betadt = (1 - exp(-beta*dt));

    // factor of infection tau=(1-exp(-beta*dt))*(d/D)^2
    tau = betadt*dratio;

    // one day in time slots
    dts = int(time_duration(24, 0, 0)/dworld().interval_td());
    // l in time slots
    lts = l*dts;
    // m in time slots
    rts = m * dts;
}


/**
 * Initialize the infected users and the other users
 */
void Infections::init_infected() {
    std::cout << "Infection::init_infected" << std::endl;

    _infections.clear();

    // initialize the '_infections' data structure
    for (const user_t& user : dworld().users())
        _infections[user].set(lts, rts);

    // scan the encounters to identify the timeslot when an infected user
    // is encountered for the first time. This timeslot MINUS the lts (latent
    // time slots) will be the 'infected' timeslot.
    bool complete = false;
    size_t n_users = dworld().users().size();
    std::unordered_set<user_t> processed;

    // t->user->{user,...}
    const tms_users &encs = dworld().get_time_encounters();
    for (auto it = encs.cbegin(); it != encs.cend() && !complete; ++it) {
        // time slot
        int t = it->first;

        const ms_users& musers = it->second;

        for (auto uit = musers.cbegin(); uit != musers.end() && !complete; ++uit) {
            const user_t& user = uit->first;

            // it is an already processed user
            if (stdx::contains(processed, user))
                continue;

            // it is an infected user
            if (stdx::contains(_infected, user))
                _infections[user].infective(t);
            else
                _infections[user].not_infected(t);

            // register the user
            processed.insert(user);
            complete = (n_users == processed.size());
        }
    }
}


void Infections::propagate_infection() {

    // t -> user -> {user,...}
    const tms_users & encs = dworld().get_time_encounters();

    //
    //  1) for each time slot
    //      2) for each users set
    //          3) apply the contact model
    //          4) compute the aggregate infection probability
    //          5) update the infections probability for each user
    //

    // previous day
    int pd = 0;

    // 1) for each time slot
    for(auto it = encs.cbegin(); it != encs.cend(); ++it) {
        int t = it->first;
        int d = t/dts;

        // if it is the new day, update the prob based on the test prob
        if (pd != d) {
            for(const user_t& user : dworld().users())
                daily_prob(t, user, test_prob());
            pd = d;
        }

        // user -> {user,...}
        const ms_users& musers = it->second;

        // 2) for each user
        for(auto uit = musers.cbegin(); uit != musers.cend(); ++uit) {
            const user_t&  u1 = uit->first;
            const s_users& uset = uit->second;

            // 3) apply the contact model
            s_users users = apply_contact_model(t, uset);

            // 4.1) for each encounter
            for(auto eit = uset.cbegin(); eit != uset.cend(); ++eit) {
                const user_t& u2 = *eit;

                if (u1 == u2) continue;

                double u1_before = _infections[u1].prob();
                double u2_prob   = _infections[u2].prob(t);
                //double u1_after  = update_prob(t, u1, tau*u2_prob*latent(_infections[u2].infected(), t));
                double u1_after  = update_prob(t, u1, tau*u2_prob);

                if (u1_before == 0 && u2_prob == 0 && u1_after != 0)
                    std::cout << "opps" << std::endl;

                if ((!_only_infections) || (u1_after != u1_before) || (u2_prob != 0.))
                    _daily_infections[d][u1].emplace_back(u1, u2, u1_after, u1_before, u2_prob);
            }

            // 4.2) compute the aggregate infection probability, excluding 'user'
            //double aprob = compute_aggregate_prob(t, u1, users);
            //
            // 5) update the infections probability for 'user'
            //update_prob(t, u1, aprob);
        }
    }
}


//double Infections::compute_aggregate_prob(int t, const user_t& user, const s_users &users) {
//    // aggregate prom
//    double ap = 1.;
//
//    for (const user_t& other : users) {
//        if (other == user)
//            continue;
//
//        // 'other' prob
//        double op = _infections[other].prob(t);
//        if (op != 0.)   // DEBUG
//            ap *= (1. - tau * op * latent(_infections[other].infected(), t));
//    }
//    return 1. - ap;
//}

double Infections::update_prob(int t, const user_t& user, double aprob) {
    return _infections[user].update(t, aprob);
}

void Infections::daily_prob(int t, const user_t& user, double aprob) {
    _infections[user].daily(t, aprob);
}

double Infections::latent(int t0, int t) const {
    if (t0 == invalid || t < t0)
        return 0.;
    int dt = t - t0;
    if (lts <= dt && dt <= rts)
        return 1.;
    else
        return 0.;
}


// --------------------------------------------------------------------------
// IO
//


static std::string with_ext(const std::string& filename, const std::string& ext) {
    int pos = filename.rfind('.');
    return filename.substr(0, pos) + ext;
}


void Infections::save_info(const std::string& filename) const {
    std::ofstream ofs(with_ext(filename, "_info.csv"));
    ofs << "name,value" << std::endl
        << "side,"<< dworld().side() << /*" m" <<*/ std::endl
        << "interval,"<< dworld().interval() << /*" min" <<*/ std::endl
        << "n_users,"<< dworld().users().size() << std::endl
        << "contact_range," << contact_range() << /*" m" <<*/ std::endl
        << "infection_rate," << infection_rate() << /*" rate/day" <<*/ std::endl
        << "cr_over_s," << (((double)contact_range())/dworld().side())  << std::endl
        << "beta," << beta  << std::endl
        << "dt," << dt  << std::endl
        << "tau,"<< tau  << std::endl
        << "latent_days," << latent_days() << /*" days" <<*/ std::endl
        << "removed_days," << removed_days() << /*" days" <<*/ std::endl
        << "n_infected," << _infected.size()  << std::endl
        << "infected," << stdx::str(_infected) << std::endl;
}


void Infections::save_table(const std::string& filename, const time_duration& interval) const {
    std::cout << "Infections::saving in " << filename << " ..." << std::endl;

    // header
    std::ofstream ofs(with_ext(filename, ".csv"));
    ofs << "day,timeslot";
    for (const user_t& user : dworld().users())
        ofs << "," << user << "";
    ofs << std::endl;

    // slots per day
    int dslots = max(1, (int)(interval/dworld().interval_td()));
    // end (latest) slot
    int eslot = (ptime(date(2020, 4, 1)) - ptime(date(2020,1,1)))/dworld().interval_td();

    for(int t=0; t < eslot; t += dslots) {
        int d = t/dts;

        ofs << d << "," << t;

        for (const user_t &user : dworld().users()) {
            double prob = _infections.at(user).prob(t);
            ofs << stdx::format(",%.5g", prob);
        }
        ofs << std::endl;
    }

    std::cout << "Infections::done" << std::endl;
}


//static void collect_daily(tmb_users& daily_encs, const tms_users& encs, int dts) {
//
//    // 1) for each time slot
//    for(auto it = encs.cbegin(); it != encs.cend(); ++it) {
//        int d = (it->first)/dts;
//        const ms_users& musers = it->second;
//
//        // 2) for each (user -> encounters)
//        for(auto uit = musers.cbegin(); uit != musers.cend(); ++uit) {
//            const user_t& user = uit->first;
//            const s_users& users = uit->second;
//
//            daily_encs[d][user].insert(users.cbegin(), users.cend());
//        }
//    }
//}

void Infections::save_daily(const std::string& filename, file_format format) const {
    if (format == file_format::CSV)
        save_daily_csv(filename);
    else
        save_daily_xml(filename);
}

void Infections::save_daily_csv(const std::string& filename) const {
    std::cout << "Infections::saving in " << filename << " ..." << std::endl;

    // ----------------------------------------------------------------------
    // IO

    std::ofstream ofs(with_ext(filename, "_daily.csv"));

    ofs << "day,u1,u2,index,u1_after,u1_before,u2_prob" << std::endl;

    //t -> u1 -> [(u1,u2,u1a, u1b, u2p), ...]

    // for all timeslots
    for(auto it = _daily_infections.cbegin(); it != _daily_infections.cend(); ++it) {
        int d = it->first;

        const users_encs_t& uencs = it->second;

        // for all users
        for(auto uit = uencs.cbegin(); uit != uencs.cend(); ++uit) {
            const user_t& u1 = uit->first;
            const user_encs_t& u1encs = uit->second;

            int i = 0;
            for (auto u1enc : u1encs) {
                ofs << d << "," << "," << u1 << "," << u1enc.u2 << "," << i++ << ","
                    << u1enc.u1_after << "," << u1enc.u1_before << "," << u1enc.u2_prob << ","
                    << std::endl;
            }

        }
    }

    std::cout << "Infections::done" << std::endl;
}


void Infections::save_daily_xml(const std::string& filename) const {
    std::cout << "Infections::saving in " << filename << " ..." << std::endl;

    // ----------------------------------------------------------------------
    // IO

    std::ofstream ofs(with_ext(filename, "_daily.xml"));

    //ofs << "day,user,encounter,prob" << std::endl;
    ofs << "<?xml version=\"1.0\" encoding=\"UTF-8\" ?>\n";
    ofs << "<infections>\n";

    // for all timeslots
    for(auto it = _daily_infections.cbegin(); it != _daily_infections.cend(); ++it) {
        int d = it->first;

        ofs << "<i d=\"" << d << "\">\n";

        const users_encs_t& uencs = it->second;

        // for all users
        for(auto uit = uencs.cbegin(); uit != uencs.cend(); ++uit) {
            const user_t& u1 = uit->first;
            const user_encs_t& u1encs = uit->second;

            ofs << "  <u1 id=\"" << u1 << "\">" << std::endl;

            int i = 0;
            for (auto u1enc : u1encs) {
                ofs << "    <u2 id=\"" << u1enc.u2 << "\" "
                    << "u1a=\""  << u1enc.u1_after  << "\" "
                    << "u1b=\"" << u1enc.u1_before << "\" "
                    << "u2p\""   << u1enc.u2_prob   << "\" "
                    << "/>"
                    << std::endl;
            }

            ofs << "  </u1>" << std::endl;
        }

        ofs << "</i>" << std::endl;
    }

    ofs << "</infections>\n";
    std::cout << "Infections::done" << std::endl;

}
