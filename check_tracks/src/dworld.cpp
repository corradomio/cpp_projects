//
// Created by Corrado Mio on 17/09/2020.
//
#include <string>
#include <iostream>
#include <stdx/to_string.h>
#include <stdx/random.h>

#include "dworld.h"

using namespace boost::gregorian;
using namespace boost::posix_time;
using namespace boost;
using namespace hls::khalifa::summer;


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

    _cusers[c].insert(user);
    _ucoords[user].push_back(c);
    _susers.insert(user);
}

// --------------------------------------------------------------------------

int merge_encs(vs_users& encs) {
    std::set<int, std::greater<int>> skip;
    bool invalid = true;
    int count = 1;

    while(invalid) {
        int n = encs.size();

        invalid = false;
        skip.clear();

        for(int i=0; i<n; ++i) {
            if (stdx::contains(skip, i)) continue;
            for(int j=i+1; j<n; ++j) {
                if (stdx::contains(skip, j)) continue;

                if (stdx::has_intersection(encs[i], encs[j])) {
                    skip.insert(j);
                    encs[i].insert(encs[j].cbegin(), encs[j].cend());
                    invalid = true;
                }
            }
        }

        count += skip.size();
        for(int i : skip)
            encs.erase(encs.begin() + i);
    }
    return count;
}


void DiscreteWorld::time_encounters() {
    for(auto it = _cusers.cbegin(); it != _cusers.cend(); ++it) {
        int t = it->first.t;
        if (it->second.size() > 1) {
            _encs[t].push_back(it->second);
        }
    }

}


void DiscreteWorld::merge_encounters() {
    if (_merged)
    for(auto it = _encs.begin(); it != _encs.end(); ++it) {
        vs_users& encs = it->second;
        merge_encs(encs);
    }
}


void DiscreteWorld::done() {
    time_encounters();
    merge_encounters();
}


// --------------------------------------------------------------------------

s_users DiscreteWorld::users(int s) const {
    stdx::random_t rnd;
    int n = _susers.size();
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
              << "      merged: " << _merged << "\n"
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

ptime DiscreteWorld::to_timestamp(int t) const {
    return ptime(_begin_time.date(), t*_interval);
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

static std::string str(const vs_users& vs) {
    std::stringstream sbuf;
    std::string sep = "|";

    sbuf << "[";
    if (!vs.empty()) {
        sbuf << stdx::str(vs[0], sep);
        for(size_t i=1; i<vs.size(); ++i)
            sbuf << ";" << stdx::str(vs[i], sep);
    }

    sbuf << "]";
    return sbuf.str();
}

static std::string str(const coords_t& c) {
    return stdx::format("%d,%d,%d", c.i, c.j, c.t);
}


void DiscreteWorld::save_slot_encounters(const std::string& filename) {
    std::cout << "DiscreteWorld::slot encounters " << filename << "[" << _cusers.size() << "]..." << std::endl;
    std::string sep = "|";

    std::ofstream ofs(filename);
    ofs << R"("latitude","longitude","timestamp","encounters")" << std::endl;

    int count = 0;
    for (auto it=this->_cusers.cbegin(); it != _cusers.end(); it++) {
        const coords_t& c = it->first;
        const s_users& users = it->second;

        ++count;
        if (count % 1000000 == 0)
            std::cout << "... " << count << "\r";

        if (users.size() > 1) {
            ofs << str(c) << ",\"" << stdx::str(users, sep) << "\"" << std::endl;
        }
    }

    std::cout << "DiscreteWorld::done" << std::endl;
}

void DiscreteWorld::save_time_encounters(const std::string& filename) {
    std::cout << "DiscreteWorld::time encounters " << filename << "[" << _encs.size() << "]..." << std::endl;

    std::ofstream ofs(filename);
    ofs << R"("timestamp","encounters")" << std::endl;

    std::vector<int> tids = stdx::keys(_encs, true);

    for (int t : tids) {
        ptime timestamp = to_timestamp(t);

        if (!_encs[t].empty())
            ofs << t << "," << "\"" << str(_encs[t]) << "\"" << std::endl;
    }

    std::cout << "DiscreteWorld::done" << std::endl;
}