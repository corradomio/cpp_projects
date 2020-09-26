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

state_t& state_t::infective() {
    _prob[0] = 1.;
    initial = true;
    return *this;
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

    if (initial)
        return p;

    int lts = inf_p->latent_days_ts();
    int rts = inf_p->removed_days_ts();

    if (rts > 0 && (t-infected) >= rts)
        return 0.;
    if (lts > 0 && (t-infected) < lts)
        return 0.;

    return p;
}


state_t& state_t::update(int t, double u) {
    int s = select(t);
    double p = (s == -1) ? 0. : _prob.at(s);
    p = 1. - (1. - p)*(1. - u);

    if (p != 0.)
        _prob[t] = p;

    // trick to set 'infected' at the correct time slot
    // for 'starting infected users'
    if (initial) {
        infected = t - inf_p->latent_days();
        initial = false;
    }
    // user just infected
    else if (p != 0 && infected == 0)
        infected = t;

    return *this;
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

    std::unordered_set<std::string> selected;

    while(n != selected.size()) {
        const user_t& user = users[rnd.next_int(nUsers)];
        selected.insert(user);
    }

    return infected(selected);
}


/// Set the list of infected ids
Infections& Infections::infected(const s_users& users) {
    _infected.insert(users.begin(), users.end());
    return *this;
}


// --------------------------------------------------------------------------

Infections& Infections::simulate(int n, double quota) {
    init();

    for(int i : stdx::range<int>(n)) {
        infected(quota);
        propagate();
    }

    return *this;
}

// --------------------------------------------------------------------------

Infections& Infections::init() {
    std::cout << "Infection::init" << std::endl;

    // cell side
    double D = dworld().side();

    // (d/D)^2
    dratio = sq(d/D);

    // time interval
    time_duration interval = dworld().interval_td();

    // DT: infection's prob in a single time slot
    double DT = interval/time_duration(24,0,0);
    // (1-exp(-beta*DT))
    betadt = (1 - exp(-beta * DT));

    // one day in time slots
    dts = int(time_duration(24, 0, 0)/interval);
    // l in time slots
    lts = l*dts;
    // m in time slots
    mts = m*dts;

    // set the parent
    for (const user_t& user : dworld().users())
        _infections[user].inf(this);

    return *this;
}


Infections& Infections::propagate() {
    std::cout << "Infection::propagate ..." << std::endl;

    const std::map<int, vs_users>& encs = dworld().get_time_encounters();

    // reset infection results
    for (const user_t& user : dworld().users())
        _infections[user].clear();
    // set initial infected
    for (const user_t& user : _infected)
        _infections[user].infective();

    int n = encs.size();
    int i = 0;
    for(auto it = encs.cbegin(); it != encs.cend(); ++it) {
        int t = it->first;
        const vs_users& usets = it->second;

        ++i;
        if (i%1000 == 0)
            std::cout << "  " << i << "/"<< n << std::endl;

        for(auto uit = usets.cbegin(); uit != usets.cend(); ++uit) {
            const s_users& users = *uit;

            double prob = compute_aggregate_prob(t, users);
            update_prob(t, users, prob);
            continue;
        }
    }

    std::cout << "Infection::end" << std::endl;
    return *this;
}


double Infections::compute_aggregate_prob(int t, const s_users &users) {
    double f = betadt*dratio;
    double ap = 1.;

    for (const std::string& user : users) {
        double up = _infections[user].prob(t);

        if (up != 0.)
            ap *= (1. - f*up);
    }
    if (ap != 1.)
        return 1. - ap;
    else
        return 0.;
}


void Infections::update_prob(int t, const s_users &users, double aprob) {
    for (const std::string& user : users) {
        _infections[user].update(t, aprob);
    }
}


// --------------------------------------------------------------------------
// IO
//

static int max(int x, int y) { return x > y ? x : y; }


void Infections::save(const std::string& filename, const time_duration& interval) const {
    std::cout << "Infections::saving in " << filename << " ..." << std::endl;

    const std::map<int, vs_users>& encs = dworld().get_time_encounters();
    std::ofstream ofs(filename);

    // comments
    {
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

    // header
    {
        ofs << "\"timestamp\"";
        for (const user_t& user : dworld().users())
            ofs << ",\"" << user.c_str() << "\"";
        ofs << std::endl;
    }

    int dslots = max(1, (int)(interval/dworld().interval_td()));
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
