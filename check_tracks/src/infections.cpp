//
// Created by Corrado Mio on 19/09/2020.
//
#include <algorithm>
#include <tbb/parallel_for_each.h>
#include <boostx/date_time_op.h>
#include "infections.h"
#include <stdx/cmath.h>
#include <stdx/containers.h>

using namespace hls::khalifa::summer;
using namespace stdx::math;
using namespace boost::posix_time;


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

state_t& state_t::prob(int t, double p) {
    if (p != 0.) {
        int s = select(t);
        if (s == -1 || _prob[s] != p) _prob[t] = p;
    }
    return *this;
}

double state_t::prob(int t) const {
    int s = select(t);
    return (s == -1) ? 0. : _prob.at(s);
}

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

    //for (const user_t& user : dworld().users())
    //    _infections[user].prob(0, 0.);
    for (const user_t& user : _infected)
        _infections[user].prob(0, 1.);

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
    time_duration interval = dworld().interval();

    // DT: infection's prob in a single time slot
    double DT = interval/time_duration(24,0,0);
    // (1-exp(-beta*delta_t))
    betadt = (1 - exp(-beta * DT));

    // time slots
    // one day in time slots
    dts = int(time_duration(24, 0, 0)/interval);
    // l in time slots
    lts = l*dts;
    // m in time slots
    mts = m*dts;

    for (const user_t& user : dworld().users())
        _infections[user].dworld(dworld());

    return *this;
}


Infections& Infections::propagate() {
    std::cout << "Infection::propagate ..." << std::endl;

    const std::map<int, vs_users>& encs = dworld().get_time_encounters();

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
    double p = 1.;

    for (const std::string& user : users) {
        double prob = _infections[user].prob(t);
        p *= (1. - f*prob);
    }
    if (p != 1.)
        return 1. - p;
    else
        return 0.;
}



void Infections::update_prob(int t, const s_users &users, double aprob) {
    for (const std::string& user : users) {
        double uprob = _infections[user].prob(t);
        double nprob = 1 - (1 - uprob)*(1 - aprob);

        _infections[user].prob(t, nprob);
    }
}

// --------------------------------------------------------------------------
// IO
//

static int max(int x, int y) { return x > y ? x : y; }


void Infections::save(const std::string& filename) const {
    std::cout << "Infections::saving in " << filename << " ..." << std::endl;

    const std::map<int, vs_users>& encs = dworld().get_time_encounters();
    std::ofstream ofs(filename);

    // header
    ofs << "\"timestamp\"";
    for (const user_t& user : dworld().users())
        ofs << ",\"" << user.c_str() << "\"";
    ofs << std::endl;

    // data
    int i = 0, n = encs.size();
    for (auto it = encs.cbegin(); it != encs.cend(); ++it) {
        int t = it->first;

        ofs << t;

        i += 1; if (i % 500 == 0)
            std::cout << "    " << stdx::format("%5d/%d", i, n) << " ..." << std::endl;

        for(const user_t& user : dworld().users()) {
            double prob = _infections.at(user).prob(t);
            ofs << stdx::format(",%.5g", prob);
        }
        ofs << std::endl;
    }

    std::cout << "Infections::done" << std::endl;
}

void Infections::save(const std::string& filename, const time_duration& interval) const {
    std::cout << "Infections::saving in " << filename << " ..." << std::endl;

    const std::map<int, vs_users>& encs = dworld().get_time_encounters();
    std::ofstream ofs(filename);

    // header
    ofs << "\"timestamp\"";
    for (const user_t& user : dworld().users())
        ofs << ",\"" << user.c_str() << "\"";
    ofs << std::endl;

    int dslots = max(1, interval/dworld().interval());
    int eslot = (ptime(date(2020, 4, 1)) - ptime(date(2020,1,1)))/dworld().interval();

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
