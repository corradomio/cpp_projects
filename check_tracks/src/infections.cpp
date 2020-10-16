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
// ustate_t
// --------------------------------------------------------------------------

double ustate_t::update(int t, double p) {
    if (_infected == invalid && _prob == 1.) {
        _infected  = t - p_inf->latent_days_ts();
        _infective = t;
        _removed   = _infected + p_inf->removed_days_ts();
    }
    else if (_infected == invalid && p > 0.) {
        _infected = t;
        _infective = t + p_inf->latent_days_ts();
        _removed   = _infected + p_inf->removed_days_ts();
        _prob = p;
    }
    else {
        _prob = 1 - (1 - _prob)*(1 - p);
    }
    return _prob;
}

ustate_t& ustate_t::tested(int t, double p) {
    _tested = t;
    _life[0] = 1 - p;
    _life[1] = p;
    return *this;
}

ustate_t& ustate_t::infected(int t, double p) {
    _tested = t;
    _life[0] = 1 - p;
    _life[1] = p;
    return *this;
}



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


//int ustate_t::select(int t) const {
//    int n = encounters.size();
//    if (n == 0)
//        return invalid;
//    else
//        encounters[n-1].t;
//}


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
    dt = dworld().interval_td()/time_duration(24,0,0);
    // (1-exp(-beta*DT))
    double betadt = (1 - exp(-beta * dt));

    // factor of infection
    tau = betadt*dratio;

    // one day in time slots
    dts = int(time_duration(24, 0, 0)/dworld().interval_td());
    // l in time slots
    lts = l*dts;
    // m in time slots
    mts = m*dts;
    // r in timeslots
    rts = r*dts;
}


void Infections::init_infected() {
    std::cout << "Infection::init_infected" << std::endl;

    std::unordered_set<user_t> processed;

    const tms_users &encs = dworld().get_time_encounters();

    _infections.clear();

    //// initialize the '_infections' data structure
    for (const user_t& user : dworld().users())
        _infections[user].inf(this);

    //// scan the encounters to identify the timeslot when an infected user
    //// is encountered for the first time. This timeslot MINUS the lts (latent
    //// time slots) will be the 'infected' timeslot.
    bool complete = false;
    for (auto it = encs.cbegin(); it != encs.cend() && !complete; ++it) {
        int t = it->first;
        const ms_users& vsusers = it->second;

        for (auto vit = vsusers.cbegin(); vit != vsusers.end() && !complete; ++vit) {
            const user_t& user = vit->first;

            // it is not an infected user
            if (!stdx::contains(_infected, user))
                continue;
            // it is an already processed user
            if (stdx::contains(processed, user))
                continue;

            // it is:
            // - an infected user
            // - a not already processed user

            int inft = t - this->latent_days_ts();
            _infections[user].infective(inft);

            // register the infected user
            processed.insert(user);
            complete = (_infected.size() == processed.size());
        }
    }

    // set the NOT infected users
    //for (const user_t& user : dworld().users())
    //    if (!stdx::contains(_infected, user))
    //        _infections[user].update(0, 0.)

    //for (const user_t& user : _infected)
    //    _infections[user].infective(0);
}


void Infections::propagate_infection() {

    const tms_users & encs = dworld().get_time_encounters();

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
        const ms_users& musers = it->second;

        // 2) for each user
        for(auto uit = musers.cbegin(); uit != musers.cend(); ++uit) {
            const user_t&  user = uit->first;
            const s_users& uset = uit->second;

            // 3) apply the contact model
            s_users users = apply_contact_model(t, uset);

            // 4.0) compute the aggregate infection probability, excluding 'user'
            double aprob = compute_aggregate_prob(t, user, users);

            // 5) update the infections probability for 'user'
            update_prob(t, user, aprob);
        }
    }
}


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
        << "side,"<< dworld().side() << " m" << std::endl
        << "interval,"<< dworld().interval() << " min" << std::endl
        << "n_users,"<< dworld().users().size() << std::endl
        << "contact_range," << contact_range() << " m" << std::endl
        << "infection_rate," << infection_rate() << " rate/day" << std::endl
        << "d/D," << (((double)contact_range())/dworld().side())  << std::endl
        << "beta," << beta  << std::endl
        << "dt," << dt  << std::endl
        << "tau,"<< tau  << std::endl
        << "latent_days," << latent_days() << " days" << std::endl
        << "removed_days," << removed_days() << " days" << std::endl
        << "n_infected," << _infected.size()  << std::endl
        << "infected," << stdx::str(_infected) << std::endl;
}


void Infections::save_table(const std::string& filename, const time_duration& interval) const {
    std::cout << "Infections::saving in " << filename << " ..." << std::endl;

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

    std::cout << "Infections::done" << std::endl;
}


static void collect_daily(tmb_users& daily_encs, const tms_users& encs, int dts) {

    // 1) for each time slot
    for(auto it = encs.cbegin(); it != encs.cend(); ++it) {
        int t = (it->first)/dts;
        const ms_users& musers = it->second;

        // 2) for each (user -> encounters)
        for(auto uit = musers.cbegin(); uit != musers.cend(); ++uit) {
            const user_t& user = uit->first;
            const s_users& users = uit->second;

            daily_encs[t][user].insert(users.cbegin(), users.cend());
        }
    }
}

void Infections::save_daily(const std::string& filename, file_format format) const {
    if (format == file_format::CSV)
        save_daily_csv(filename);
    else
        save_daily_xml(filename);
}

void Infections::save_daily_csv(const std::string& filename) const {
    std::cout << "Infections::saving in " << filename << " ..." << std::endl;

    // initialize the user infections probability
    std::unordered_map<user_t, ustate_t> uprobs;
    for(const user_t& user : dworld().users())
        uprobs[user].inf(*this);
    for(const user_t& user : _infected)
        uprobs[user].infected();

    // ----------------------------------------------------------------------
    // IO

    std::ofstream ofs(with_ext(filename, "_daily.csv"));

    const tms_users& encs = dworld().get_time_encounters();
    tmb_users daily_encs;

    ofs << "day,user,encounter,prob" << std::endl;

    // collect encounters
    collect_daily(daily_encs, encs, dts);

    // for all timestamps
    for(auto it = daily_encs.cbegin(); it != daily_encs.cend(); ++it) {
        int t = it->first;

        const std::unordered_map<user_t, b_users>& users = it->second;

        // for all users
        for(auto uit = users.cbegin(); uit != users.cend(); ++uit) {
            const user_t& u1 = uit->first;
            const b_users& eusers = uit->second;

            // for all encountered users
            for(auto eit = eusers.cbegin(); eit != eusers.cend(); ++eit) {
                const user_t& u2 = eit->first;
                int count = eit->second;

                double pu2 = uprobs[u2].prob(t);
                double np1 = uprobs[u1].update(t, pow((1 - tau*pu2), count));

                ofs << t << "," << u1 << "," << u2 << "," << np1 << std::endl;
            }
        }
    }

    std::cout << "Infections::done" << std::endl;
}


void Infections::save_daily_xml(const std::string& filename) const {
    std::cout << "Infections::saving in " << filename << " ..." << std::endl;

    // initialize the user infections probability
    std::unordered_map<user_t, ustate_t> uprobs;
    for(const user_t& user : dworld().users())
        uprobs[user].inf(*this);
    for(const user_t& user : _infected)
        uprobs[user].infected();

    // ----------------------------------------------------------------------
    // IO

    std::ofstream ofs(with_ext(filename, "_daily.xml"));

    const tms_users& encs = dworld().get_time_encounters();
    tmb_users daily_encs;

    //ofs << "day,user,encounter,prob" << std::endl;
    ofs << "<?xml version=\"1.0\" encoding=\"UTF-8\" ?>\n";
    ofs << "<infections>\n";

    // collect encounters
    collect_daily(daily_encs, encs, dts);

    // for all timestamps
    for(auto it = daily_encs.cbegin(); it != daily_encs.cend(); ++it) {
        int t = it->first;

        ofs << "<infection day=\"" << t << "\">\n";

        const std::unordered_map<user_t, b_users>& users = it->second;

        // for all users
        for(auto uit = users.cbegin(); uit != users.cend(); ++uit) {
            const user_t& u1 = uit->first;
            const b_users& eusers = uit->second;

            ofs << "  <user id=\"" << u1 << "\">\n";

            // for all encountered users
            for(auto eit = eusers.cbegin(); eit != eusers.cend(); ++eit) {
                const user_t& u2 = eit->first;
                int count = eit->second;

                double pu2 = uprobs[u2].prob(t);
                double np1 = uprobs[u1].update(t, pow((1 - tau*pu2), count));

                ofs << "    <other id=\"" << u2 << "\"  prob=\"" << np1 << "\" />\n";
            }

            ofs << "  </user>\n";
        }

        ofs << "</infection>\n";
    }

    ofs << "</infections>\n";
    std::cout << "Infections::done" << std::endl;
}
