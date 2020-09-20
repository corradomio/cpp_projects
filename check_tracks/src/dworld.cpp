//
// Created by Corrado Mio on 17/09/2020.
//
#include <string>
#include <iostream>
#include <csvstream.h>

#include "dworld.h"

using namespace boost::filesystem;
using namespace boost::gregorian;
using namespace boost::posix_time;
using namespace boost;
using namespace hls::khalifa::summer;


const std::string DATASET = R"(D:\Dropbox\2_Khalifa\Progetto Summer\Dataset_3months)";


// ----------------------------------------------------------------------
//
// ----------------------------------------------------------------------

void DiscreteWorld::add(const std::string& id, double latitude, double longitude, const ptime& timestamp) {
    coords_t loc = to_coords(latitude, longitude, timestamp);

    _sdata.add(loc, id);
    _udata.add(id, loc);
}

// ----------------------------------------------------------------------

const v_users& DiscreteWorld::ids() const {
    if (_users.empty()) {
        for (auto it = _udata._data.begin(); it != _udata._data.end(); it++)
            _users.push_back(it->first);
    }
    return _users;
}

std::map<int, encounters_set_t> DiscreteWorld::get_encounters(const std::string& id) const {
    std::map<int, encounters_set_t> encs;

    const std::vector<coords_t>& ucoords = _udata[id];
    for (const coords_t& c : ucoords) {
        const std::unordered_set<std::string>& sdata = _sdata[c];

        encs[c.t].add(c, sdata);
    }

    std::vector<int> tv = stdx::keys(encs);
    for (int t : tv)
        encs[t].eids.erase(id);

    return encs;
}

// ----------------------------------------------------------------------


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

// ------------------------------------------------------------------

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

// ----------------------------------------------------------------------

void DiscreteWorld::dump() {
    std::cout << "DiscreteWorld(" << _side << "," << _interval.minutes() << ")\n"
              << "  one_degree: " << _onedegree << "\n"
              << "  begin_time: " << to_simple_string(_begin_time) << "\n"
              << "  sparse_data:" << _sdata.size() << "\n"
              << "  user_coords:" << _udata.size() << "\n"
              << "end" << std::endl;
}

// ----------------------------------------------------------------------
//
// ----------------------------------------------------------------------

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


