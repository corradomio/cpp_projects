//
// Created by Corrado Mio on 19/09/2020.
//
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
Infections& Infections::set_infected(float quota) {
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
    this->init_infections();
    return *this;
}


void Infections::init_infections() {

    for(const std::string& id : infected)
        infections[id].push_back(state_t(0, 1.));

    for(const std::string& id : dworld().ids())
        if (!stdx::contains(infected, id))
            infections[id].push_back(state_t(0, 0.));
}


