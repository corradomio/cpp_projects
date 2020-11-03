//
// Created by Corrado Mio on 17/09/2020.
//
#include <string>
#include <iostream>
#include <stdx/to_string.h>
#include <stdx/random.h>

#include "dworld.h"
#include <stdx/containers.h>

using namespace boost::gregorian;
using namespace boost::posix_time;
using namespace boost;
using namespace hls::khalifa::summer;


// --------------------------------------------------------------------------
// location_t
// --------------------------------------------------------------------------

location_t::location_t():
    min(+infty,+infty,+infty),
    max(-infty,-infty,-infty),
    first(+infty,+infty,+infty),
    last(-infty,-infty,-infty)
{ }


void location_t::update(const coords_t& c) {
    // c'e' un problema con un utente che ha coordinate negative.
    // Lo saltiamo
    if (c.i < 0 || c.j< 0)
        return;

    if (c.i < min.i) min.i = c.i;
    if (c.j < min.j) min.j = c.j;
    if (c.t < first.t) first = c;

    if (c.i > max.i) max.i = c.i;
    if (c.j > max.j) max.j = c.j;
    if (c.t > last.t) last = c;
}


// --------------------------------------------------------------------------
// DiscreteWorld
// --------------------------------------------------------------------------

DiscreteWorld::DiscreteWorld() {
    this->side(100);
    this->interval(5);
}


DiscreteWorld::DiscreteWorld(int side, int minutes) {
    this->side(side);
    this->interval(minutes);
}


DiscreteWorld::~DiscreteWorld() {
    _susers.clear();
    _cusers.clear();
    _ucoords.clear();
    //_encs.clear();
}


// --------------------------------------------------------------------------

DiscreteWorld& DiscreteWorld::side(int side) {
    _side  = side;
    _angle = _side / _onedegree;
    return *this;
}


DiscreteWorld& DiscreteWorld::interval(int minutes) {
    if (minutes == 0)
        _interval = time_duration(0, 0, 5);
    else
        _interval = time_duration(0, minutes, 0);
    return *this;
}

// --------------------------------------------------------------------------

void DiscreteWorld::add(const user_t& user, double latitude, double longitude, const ptime& timestamp) {
    coords_t c = to_coords(latitude, longitude, timestamp);

    // add user in cell (i,j,t)
    _cusers[c].insert(user);

    // add cell in the list of cells visited by user
    _ucoords[user].push_back(c);

    // add user in the set of users
    _susers.insert(user);
}

// --------------------------------------------------------------------------

void DiscreteWorld::done() {
    // compute the user encoutenred by each user in each timeslot

    _encs.clear();

    // for each cell
    // (i,j,t) -> {user,...}
    for(auto it = _cusers.cbegin(); it != _cusers.cend(); ++it) {
        int t = it->first.t;

        // for each users in the cell (with timestamp t)
        const s_users& users = it->second;
        for (const user_t& u1 : users)
        for (const user_t& u2 : users)
        if(u1 != u2) {
            _encs[t][u1].insert(u2);
            _encs[t][u2].insert(u1);
        }
    }
}


// --------------------------------------------------------------------------

s_users DiscreteWorld::users(int s) const {
    stdx::random_t rnd;
    int n = _susers.size();

    if (s >= n)
        return _susers;

    s_users selected;
    std::vector<user_t> vusers(_susers.begin(), _susers.end());

    while (selected.size() != s) {
        selected.insert(vusers[rnd.next_int(n)]);
    }

    return selected;
}


// --------------------------------------------------------------------------

void DiscreteWorld::dump() {
    std::cout << "DiscreteWorld(" << side() << "," << interval() << ")\n"
              << "  one_degree: " << _onedegree << "\n"
              << "  begin_time: " << to_simple_string(_begin_time) << "\n"
              << "        side: " << _side << "\n"
              << "    interval: " << interval() << "\n"
              << "       angle: " << _angle << "\n"
              << "       users: " << _susers.size() << "\n"
              << "  user_cells: " << _cusers.size() << "\n"
              << " user_coords: " << _ucoords.size() << "\n"
              << "  encounters: " << _encs.size() << "\n"
              << "end" << std::endl;
}


// --------------------------------------------------------------------------
// Conversions
// --------------------------------------------------------------------------

coords_t DiscreteWorld::to_coords(double latitude, double longitude, const ptime& timestamp) {
    int i = int(latitude  / _angle);
    int j = int(longitude / _angle);
    int t = int((timestamp - _begin_time).total_seconds() / _interval.total_seconds());

    return coords_t(i, j, t);
}

//ptime DiscreteWorld::to_timestamp(int t) const {
//    return ptime(_begin_time.date(), t*_interval);
//}


// --------------------------------------------------------------------------
// locations
// --------------------------------------------------------------------------

void DiscreteWorld::update_daily_locations() {
    if (!_locs.empty())
        return;

    int dts = day_ts();

    for (auto uit = _ucoords.cbegin(); uit != _ucoords.cend(); ++uit) {
        const user_t& u = uit->first;

        const std::vector<coords_t> & coords = uit->second;

        for(const coords_t& c : coords) {
            int d = c.t/dts;

            _locs[d][u].update(c);
            _locs[-1][-1].update(c);
        }
    }
}


// --------------------------------------------------------------------------
// IO
// --------------------------------------------------------------------------

void DiscreteWorld::save(const std::string& filename) const {
    std::cout << "DiscreteWorld::saving in " << filename << " ..." << std::endl;

    std::ofstream ofs(filename, std::ios::binary);
    cereal::BinaryOutputArchive oa(ofs);
    oa << *this;

    std::cout << "DiscreteWorld::done" << std::endl;
}

void DiscreteWorld::load(const std::string& filename) {
    std::cout << "DiscreteWorld::load " << filename << " ..." << std::endl;

    std::ifstream ifs(filename, std::ios::binary);
    cereal::BinaryInputArchive ia(ifs);
    ia >> *this;

    std::cout << "DiscreteWorld::done" << std::endl;
}


// --------------------------------------------------------------------------
// Special saves
// --------------------------------------------------------------------------

//static std::string str(const vs_users& vs) {
//    std::stringstream sbuf;
//    std::string sep = "|";
//
//    sbuf << "[";
//    if (!vs.empty()) {
//        sbuf << stdx::str(vs[0], sep);
//        for(size_t i=1; i<vs.size(); ++i)
//            sbuf << ";" << stdx::str(vs[i], sep);
//    }
//
//    sbuf << "]";
//    return sbuf.str();
//}

static std::string str(const coords_t& c) {
    return stdx::format("%d,%d,%d", c.i, c.j, c.t);
}


void DiscreteWorld::save_slot_encounters(const std::string& filename) {
    std::cout << "DiscreteWorld::slot_encounters " << filename << "[" << _cusers.size() << "]..." << std::endl;
    std::string sep = "|";

    std::ofstream ofs(filename);
    ofs << "latitude,longitude,timestamp,encounters" << std::endl;

    for (auto it=this->_cusers.cbegin(); it != _cusers.end(); it++) {
        const coords_t& c = it->first;
        const s_users& users = it->second;

        if (users.size() > 1) {
            ofs << str(c) << "," << stdx::str(users, sep) << "" << std::endl;
        }
    }

    std::cout << "DiscreteWorld::slot_encounters::done" << std::endl;
}


void DiscreteWorld::save_time_encounters(const std::string& filename, bool as_set) {
    std::cout << "DiscreteWorld::time_encounters " << filename << "[" << _encs.size() << "]..." << std::endl;

    int dts = day_ts();

    std::ofstream ofs(filename);
    ofs << "day,timestamp,user,encounter" << std::endl;

    // for all times
    for(auto it = _encs.cbegin(); it != _encs.cend(); ++it) {
        int t = it->first;

        const ms_users& msusers = it->second;

        // for all users
        for (auto uit = msusers.cbegin(); uit != msusers.cend(); ++uit) {

            const user_t& user = uit->first;
            const s_users& encounters = uit->second;

            if (as_set) {
                ofs << (t/dts) << "," << t << "," << user << "," << stdx::str(encounters) << std::endl;
            }
            else {
                for (auto encounter : encounters)
                    ofs << (t/dts) << "," << t << "," << user << "," << encounter << std::endl;
            }
        }
    }

    std::cout << "DiscreteWorld::time_encounters::done" << std::endl;
}


void DiscreteWorld::save_daily_locations(const std::string& filename){
    update_daily_locations();

    std::ofstream ofs(filename);
    ofs << "day,user,min_i,min_j,max_i,max_j,first_i,first_j,last_i,last_j" << std::endl;
    for (auto dit = _locs.cbegin(); dit != _locs.cend(); ++dit) {
        int d = dit->first;

        const ml_users& lusers = dit->second;
        for(auto uit = lusers.cbegin(); uit != lusers.cend(); ++uit) {
            const user_t& u = uit->first;
            const location_t& l = uit->second;

            ofs << d << "," << u
                << "," << l.min.i << "," << l.min.j
                << "," << l.max.i << "," << l.max.j
                << "," << l.first.i << "," << l.first.j
                << "," << l.last.i << "," << l.last.j
                << std::endl;
        }
    }
}