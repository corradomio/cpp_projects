//
// Created by Corrado Mio on 15/09/2020.
//

#ifndef CHECK_TRACKS_DWORLD_H
#define CHECK_TRACKS_DWORLD_H

#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <memory>

#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/date_time.hpp>
#include <boostx/date_time_op.h>

#include <cereal/types/map.hpp>
#include <cereal/types/set.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/unordered_set.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/string.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>

#include <stdx/keys.h>
#include <stdx/format.h>


using namespace boost::posix_time;
using namespace boost::gregorian;


namespace hls {
namespace khalifa {
namespace summer {

    struct coords_t {
        int i, j, t;

        coords_t():i(0), j(0), t(0) { }
        coords_t(int i, int j, int t):i(i),j(j),t(t){}
        coords_t(const coords_t& c): i(c.i), j(c.j), t(c.t){}

        //coords_t& operator =(const coords_t& c) {
        //    this->i = c.i;
        //    this->j = c.j;
        //    this->t = c.t;
        //    return *this;
        //}

        bool operator <(const coords_t& c) const {
            if (i < c.i) return true;
            if (i > c.i) return false;
            if (j < c.j) return true;
            if (j > c.j) return false;
            if (t < c.t) return true;
            return false;
        }

        bool operator ==(const coords_t& c) const {
            return i == c.i && j == c.j && t == c.t;
        }

        coords_t& operator +=(const coords_t& c) {
            i += c.i;
            j += c.j;
            t += c.t;
            return *this;
        }

        coords_t operator /(int scalar) const {
            coords_t c;
            c.i = i/scalar;
            c.j = j/scalar;
            c.t = t/scalar;
            return c;
        }

        std::string str() const {
            char buffer[128];
            sprintf(buffer, "(%d,%d,%d)", i,j,t);
            return buffer;
        }
    };

    struct coords_hash {
        size_t operator()(const coords_t& c) const
        {
            return ((c.i * 17) ^ c.j) * 17 ^ c.t;
        }
    };

    template<class Archive>
    void serialize(Archive & ar, coords_t& c)
    {
        ar(c.i, c.j, c.t);
    }

    struct dwpoint_t {
        double latitude;
        double longitude;
        ptime timestamp;

        std::string str() const {
            char buffer[256];
            sprintf(buffer, "%.06f,%.06f,\"%s\",\"%s\"", latitude, longitude,
                    to_iso_extended_string(timestamp.date()).c_str(),
                    to_simple_string(timestamp.time_of_day()).c_str());
            return buffer;
        }
    };

    // ----------------------------------------------------------------------

    struct encounters_set_t {
        int count;
        coords_t sum;
        std::unordered_set<std::string> eids;

        void add(const coords_t& c, const std::unordered_set<std::string>& ids) {
            sum += c;
            count += 1;
            eids.insert(ids.begin(), ids.end());
        }

        coords_t get_coords() const {
            return sum/count;
        }
    };

    // ----------------------------------------------------------------------

    struct sparse_data_t {
        // (i,j,t) -> {id}
        std::unordered_map<coords_t, std::unordered_set<std::string>, coords_hash> _data;
        std::unordered_set<std::string> _empty;

        const std::unordered_set<std::string>& operator[](const coords_t& loc) const {
            if (_data.find(loc) == _data.end())
                return _empty;
            else
                return _data.at(loc);
        }

        void add(const coords_t& loc, const std::string& id) {
            _data[loc].insert(id);
        }

        size_t size() const {
            return _data.size();
        }
    };

    struct user_coords_t {
        // id -> [(i,j,t), ...]
        std::unordered_map<std::string, std::vector<coords_t>> _data;

        const std::vector<coords_t>& operator[](const std::string& id) const {
            return const_cast<std::unordered_map<std::string, std::vector<coords_t>>&>(_data)[id];
        }

        void add(const std::string& id, const coords_t& coords) {
            _data[id].push_back(coords);
        }

        size_t size() const {
            return _data.size();
        }
    };

    typedef std::vector<std::string> users_t;

    // ----------------------------------------------------------------------

    struct DiscreteWorld {
        /// length of 1 degree (in meters)
        double _onedegree = 111319;
        /// first day
        ptime  _begin_time = ptime(date(2020,1,1));

        /// grid side (in meters)
        double _side;

        /// time interval
        time_duration _interval;

        /// side in degrees
        double _angle;

        coords_t to_coords(double latitude, double longitude, const ptime& timestamp) {
            int i = int(latitude  / _angle);
            int j = int(longitude / _angle);
            int t = int((timestamp - _begin_time).total_seconds() / _interval.total_seconds());

            return coords_t(i, j, t);
        }

        mutable users_t _users;
        sparse_data_t _sdata;
        user_coords_t _ucoords;

    public:
        DiscreteWorld() {
            this->side(100);
            this->interval(5);
        }

        DiscreteWorld(double side, int minutes) {
            this->side(side);
            this->interval(minutes);
        }

        //
        // Set parameters
        //

        DiscreteWorld& side(double side) {
            _side  = side;
            _angle = _side / _onedegree;
            return *this;
        }

        DiscreteWorld& interval(int minutes) {
            if (minutes == 0)
                _interval = time_duration(0, 0, 5);
            else
                _interval = time_duration(0, minutes, 0);
            return *this;
        }

        //
        // Populate
        //

        void add(const std::string& id, double latitude, double longitude, const ptime& timestamp) {
            coords_t loc = to_coords(latitude, longitude, timestamp);

            _sdata.add(loc, id);
            _ucoords.add(id, loc);
        }

        // ----------------------------------------------------------------------

        const std::vector<std::string>& get_ids() const {
            if (_users.empty()) {
                for (auto it = _ucoords._data.begin(); it != _ucoords._data.end(); it++)
                    _users.push_back(it->first);
            }

            return _users;
        }

        ///
        /// \param id
        /// \return  map[t -> set[id]]
        std::map<int, encounters_set_t> get_encounters(const std::string& id) const {
            std::map<int, encounters_set_t> encs;

            const std::vector<coords_t>& ucoords = _ucoords[id];
            for (const coords_t& c : ucoords) {
                const std::unordered_set<std::string>& sdata = _sdata[c];

                encs[c.t].add(c, sdata);
            }

            std::vector<int> tv = stdx::keys(encs);
            for (int t : tv)
                encs[t].eids.erase(id);

            return encs;
        }

        ptime to_timestamp(int t) const {
            return ptime(_begin_time.date(), t*_interval);
        }

        dwpoint_t to_point(const coords_t& c) const {
            dwpoint_t p;
            p.latitude  = c.i*_angle;
            p.longitude = c.j*_angle;
            p.timestamp = to_timestamp(c.t);
            return p;
        }

        // ----------------------------------------------------------------------

        void dump() {
            std::cout << "DiscreteWorld(" << _side << "," << _interval.minutes() << ")\n"
            << "  one_degree: " << _onedegree << "\n"
            << "  begin_time: " << to_simple_string(_begin_time) << "\n"
            << "  sparse_data:" << _sdata.size() << "\n"
            << "  user_coords:" << _ucoords.size() << "\n"
            << "end" << std::endl;
        }

        // ------------------------------------------------------------------

        void save(const std::string filename) {
            std::cout << "Saving in " << filename << " ..." << std::endl;

            std::ofstream ofs(filename, std::ios::binary);
            cereal::BinaryOutputArchive oa(ofs);
            oa << *this;

            std::cout << "  done" << std::endl;
        }

        void load(const std::string filename) {
            std::cout << "Loading " << filename << " ..." << std::endl;

            std::ifstream ifs(filename, std::ios::binary);
            cereal::BinaryInputArchive ia(ifs);
            ia >> *this;

            std::cout << "  done" << std::endl;
        }

        template<class Archive>
        void save(Archive & ar) const
        {
            int seconds = _interval.total_seconds();
            ar(_side, _angle, seconds, _sdata._data, _ucoords._data);
        }
        template<class Archive>
        void load(Archive & ar)
        {
            int seconds;
            ar(_side, _angle, seconds, _sdata._data, _ucoords._data);
            _interval = time_duration(0, 0, seconds);
        }
    };

    // ----------------------------------------------------------------------

} } }

extern void create_grid(int side, int interval, const std::string& path);

extern void load_grid(hls::khalifa::summer::DiscreteWorld& dworld, const std::string& fname);

#endif //CHECK_TRACKS_DWORLD_H
