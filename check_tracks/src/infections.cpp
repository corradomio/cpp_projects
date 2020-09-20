//
// Created by Corrado Mio on 19/09/2020.
//
#include <tbb/parallel_for_each.h>
#include <boostx/date_time_op.h>
#include <boost/date_time.hpp>
#include "infections.h"
#include <stdx/cmath.h>
#include <stdx/containers.h>

using namespace hls::khalifa::summer;
using namespace stdx::math;
using namespace boost::posix_time;


Infections& Infections::set_dworld(const DiscreteWorld& dworld_) {
    (*this).dworld_p = &dworld_;

    // cell side
    double side = dworld().side();

    // time interval
    time_duration interval = dworld().interval();

    // (d/D)^2
    double D = side;
    dratio = sq(d/D);

    // (1-exp(-beta*delta_t))
    double DT = interval/time_duration(24,0,0);
    betadt = (1 - exp(-beta * DT));

    // intervals in time slots
    oneday = int(time_duration(24, 0, 0)/interval);
    start = l*oneday;
    reset = m*oneday;

    return *this;
}


/// Quota [0,1] of infected ids
Infections& Infections::set_infected(double quota) {
    int n = int(quota*dworld().ids().size());
    return set_infected(n);
}


/// Number of infected ids
Infections& Infections::set_infected(int n) {
    const std::vector<std::string>& ids = dworld().ids();
    int nids = ids.size();

    std::unordered_set<std::string> selected;

    while(n != selected.size()) {
        std::string id = ids[rnd.next_int(nids)];
        selected.insert(id);
    }

    return set_infected(selected);
}


/// Select the list of infected ids
Infections& Infections::set_infected(const std::unordered_set<std::string>& ids) {
    infected.insert(ids.begin(), ids.end());

    for(const std::string& id : infected)
        infections[id].push_back(state_t(0, 1.));
    for(const std::string& id : dworld().ids())
        if (!stdx::contains(infected, id))
            infections[id].push_back(state_t(0, 0.));
    return *this;
}


void Infections::propagate() {
    ref::unordered_map<int, vs_users> encs = get_all_encounters();

    std::cout << encs.size() << std::endl;
}


void collapse(vs_users& users) {
    bool removed = true;

    while (removed) {
        int n = users.size();
        std::set<int, std::greater<>> toremove;

        for (int i = 0; i < n; ++i) {
            if (stdx::contains(toremove, i)) continue;

            for (int j = i + 1; j < n; ++j) {
                if (stdx::contains(toremove, j)) continue;

                if (stdx::has_intersection(users[i], users[j])) {
                    stdx::merge(users[i], users[j]);
                    toremove.insert(j);
                }
            }
        }

        removed = !toremove.empty();
        if (removed)
        for(auto it=toremove.cbegin(); it != toremove.cend(); ++it)
            users.erase(users.cbegin()+(*it));
    }

    return;
}

ref::unordered_map<int, vs_users>
Infections::get_all_encounters() {
    ref::unordered_map<int, vs_users> encs;
    std::unordered_set<int> keys;

    int n = dworld()._sdata._data.size(), i=1;
    for (auto it = dworld()._sdata._data.begin(); it != dworld()._sdata._data.end(); ++it, ++i) {
        if (i%100000 == 0)
            std::cout << "... " << i << "/" << n << std::endl;
        int t = it->first.t;
        keys.insert(t);
        encs[t].push_back(it->second);
        collapse(encs[t]);
    }

    return encs;
}


