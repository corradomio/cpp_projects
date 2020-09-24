//
// Created by Corrado Mio on 17/09/2020.
//
#include <string>
#include <iostream>
#include <csvstream.h>
#include <stdx/to_string.h>

#include "dworld.h"

using namespace boost::filesystem;
using namespace boost::gregorian;
using namespace boost::posix_time;
using namespace boost;
using namespace hls::khalifa::summer;


const std::string DATASET = R"(D:\Dropbox\2_Khalifa\Progetto Summer\Dataset_3months)";


// --------------------------------------------------------------------------
// Generic functions
// --------------------------------------------------------------------------

void create_grid(int side, int interval, const std::string& fname) {

    DiscreteWorld dworld(side, interval);

    std::cout << "create_grid(" << side << "," << interval << ")" << std::endl;

    try {
        int count = 0;

        path p(DATASET);

        // 0,  1          2           3    4          5                  6      7      8           9
        // "","latitude","longitude","V3","altitude","date.Long.format","date","time","person.id","track.id"

        for (directory_entry &de : directory_iterator(p)) {
            if (de.path().extension() != ".csv")
                continue;

            //std::cout << "  " << de.path().string()<< std::endl;

            csvstream csvin(de.path().string());

            std::vector<std::string> row;

            while  (csvin >> row) {
                count += 1;

                //if (count%100000 == 0)
                //    std::cout << "    " << count << std::endl;

                std::string id = row[8];
                double latitude  = lexical_cast<double>(row[1]);
                double longitude = lexical_cast<double>(row[2]);

                date date = from_string(row[6]);
                time_duration duration = duration_from_string(row[7]);
                ptime timestamp(date, duration);

                dworld.add(id, latitude, longitude, timestamp);
            }
        }
        std::cout << "    " << count << std::endl;

        std::cout << "save in(" << fname << ")" << std::endl;
        dworld.save(fname);
        dworld.dump();
        std::cout << std::endl;
    }
    catch(std::exception& e) {
        std::cout << e.what() << std::endl;
    }

}


void load_grid(DiscreteWorld& dworld, const std::string& fname) {
    std::cout << "load from(" << fname << ")" << std::endl;
    dworld.load(fname);
    dworld.dump();
}


// --------------------------------------------------------------------------
// DiscreteWorld
// --------------------------------------------------------------------------

void DiscreteWorld::add(const std::string& id, double latitude, double longitude, const ptime& timestamp) {
    coords_t loc = to_coords(latitude, longitude, timestamp);

    _sdata.add(loc, id);
    _udata.add(id, loc);
}


DiscreteWorld::~DiscreteWorld() {
    _sdata._data.clear();
    _udata._data.clear();
}


// --------------------------------------------------------------------------

const v_users& DiscreteWorld::ids() const {
    if (_users.empty()) {
        for (auto it = _udata._data.begin(); it != _udata._data.end(); it++)
            _users.push_back(it->first);
    }
    return _users;
}

//void DiscreteWorld::get_encounters(std::map<int, encounters_set_t>& encs, const std::string& id) const {
//    const std::vector<coords_t>& ucoords = _udata[id];
//    for (const coords_t& c : ucoords) {
//        const std::unordered_set<std::string>& sdata = _sdata[c];
//
//        encs[c.t].add(c, sdata);
//    }
//
//    std::vector<int> tv = stdx::keys(encs);
//    for (int t : tv)
//        encs[t].eids.erase(id);
//}


void merge_encs(vs_users& encs) {
    std::set<int, std::greater<int>> skip;
    bool invalid = true;

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

        for(int i : skip)
            encs.erase(encs.begin() + i);
    }

}


void DiscreteWorld::get_time_encounters(std::map<int, vs_users>& encs) const {
    std::set<int> tids;

    for(auto it = _sdata._data.cbegin(); it != _sdata._data.cend(); ++it) {
        int t = it->first.t;
        tids.insert(t);
        if (it->second.size() > 1) {
            encs[t].push_back(it->second);
        }
    }

    for(int t : tids) {
        merge_encs(encs[t]);
    }
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

dwpoint_t DiscreteWorld::to_point(const coords_t& c) const {
    dwpoint_t p;
    p.latitude  = c.i*_angle;
    p.longitude = c.j*_angle;
    p.timestamp = to_timestamp(c.t);
    return p;
}

// --------------------------------------------------------------------------
// IO
// --------------------------------------------------------------------------

void DiscreteWorld::save(const std::string filename) const {
    std::cout << "Saving in " << filename << " ..." << std::endl;

    std::ofstream ofs(filename, std::ios::binary);
    cereal::BinaryOutputArchive oa(ofs);
    oa << *this;

    std::cout << "  done" << std::endl;
}

void DiscreteWorld::load(const std::string filename) {
    std::cout << "Loading " << filename << " ..." << std::endl;

    std::ifstream ifs(filename, std::ios::binary);
    cereal::BinaryInputArchive ia(ifs);
    ia >> *this;

    std::cout << "  done" << std::endl;
}

// --------------------------------------------------------------------------

void DiscreteWorld::dump() {
    std::cout << "DiscreteWorld(" << _side << "," << _interval.minutes() << ")\n"
              << "  one_degree: " << _onedegree << "\n"
              << "  begin_time: " << to_simple_string(_begin_time) << "\n"
              << "  sparse_data:" << _sdata.size() << "\n"
              << "  user_coords:" << _udata.size() << "\n"
              << "end" << std::endl;
}


// --------------------------------------------------------------------------
// Special saves
// --------------------------------------------------------------------------

void DiscreteWorld::save_slot_encounters(const std::string filename) {
    std::cout << "slot encounters " << filename << "[" << this->_sdata._data.size() << "]..." << std::endl;

    std::ofstream ofs(filename);
    ofs << R"("latitude","longitude","timestamp","encounters")" << std::endl;

    int count = 0;
    for (auto it=this->_sdata._data.begin(); it != this->_sdata._data.end(); it++) {
        const coords_t &c = it->first;
        const s_users  &u = it->second;

        ++count;
        if (count % 1000000 == 0)
            std::cout << "... " << count << "\r";

        if (u.size() > 1) {
            //dwpoint_t pt = to_point(coords);
            ofs << c.i << "," << c.j << "," << c.t << ",\"" << stdx::str(u, "|") << "\"" << std::endl;
        }
    }
}

//void DiscreteWorld::save_slot_encounters(const std::string filename) {
//    std::cout << "slot encounters " << filename << "[" << this->_sdata._data.size() << "]..." << std::endl;
//
//    std::ofstream ofs(filename);
//    ofs << R"("latitude","longitude","date","time","encounters")" << std::endl;
//
//    int count = 0;
//    for (auto it=this->_sdata._data.begin(); it != this->_sdata._data.end(); it++) {
//        const coords_t &coords = it->first;
//        const s_users &users = it->second;
//
//        ++count;
//        if (count%1000000 == 0)
//            std::cout << "... " << count << "\r";
//
//        if (users.size() > 1) {
//            dwpoint_t pt = to_point(coords);
//            ofs << pt.str() << ",\"" << stdx::str(users, "|") << "\"" << std::endl;
//        }
//    }
//
//    std::cout << "done" << std::endl;
//}


static std::string to_str(const vs_users& vs) {
    std::stringstream sbuf;

    sbuf << "[";
    if (!vs.empty()) {
        sbuf << stdx::str(vs[0], "|");
        for(size_t i=1; i<vs.size(); ++i)
            sbuf << ";" << stdx::str(vs[i], "|");
    }

    sbuf << "]";
    return sbuf.str();
}


//void DiscreteWorld::save_time_encounters(const std::string filename) {
//    std::cout << "time encounters " << filename << "[" << this->_sdata._data.size() << "]..." << std::endl;
//
//    std::ofstream ofs(filename);
//    ofs << R"("date","time","encounters")" << std::endl;
//
//    std::map<int, vs_users> encs;
//    get_time_encounters(encs);
//
//    std::vector<int> tids = stdx::keys(encs, true);
//
//    for (int t : tids) {
//        ptime timestamp = to_timestamp(t);
//
//        ofs << "\"" << to_iso_extended_string(timestamp.date()).c_str() << "\","
//            << "\"" << to_simple_string(timestamp.time_of_day()).c_str()   << "\","
//            << "\"" << to_str(encs[t]) << "\"" << std::endl;
//
//    }
//
//    std::cout << "done" << std::endl;
//}

void DiscreteWorld::save_time_encounters(const std::string filename) {
    std::cout << "time encounters " << filename << "[" << this->_sdata._data.size() << "]..." << std::endl;

    std::map<int, vs_users> encs;
    get_time_encounters(encs);

    std::ofstream ofs(filename);
    ofs << R"("timestamp","encounters")" << std::endl;

    std::vector<int> tids = stdx::keys(encs, true);

    for (int t : tids) {
        ptime timestamp = to_timestamp(t);

        if (!encs[t].empty())
            ofs << t << "," << "\"" << to_str(encs[t]) << "\"" << std::endl;
    }

    std::cout << "done" << std::endl;
}