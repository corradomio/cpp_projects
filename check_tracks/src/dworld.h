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

#include <stdx/containers.h>
#include <stdx/format.h>


using namespace boost::posix_time;
using namespace boost::gregorian;


namespace hls {
namespace khalifa {
namespace summer {

    typedef std::vector<std::string> v_users;
    typedef std::unordered_set<std::string> s_users;
    typedef std::vector<s_users> vs_users;

    // ----------------------------------------------------------------------

    struct coords_t {
        int i, j, t;

        coords_t():i(0), j(0), t(0) { }
        coords_t(int i, int j, int t):i(i),j(j),t(t){}
        coords_t(const coords_t& c): i(c.i), j(c.j), t(c.t){}

        coords_t& operator =(const coords_t& c) {
            this->i = c.i;
            this->j = c.j;
            this->t = c.t;
            return *this;
        }

        //bool operator <(const coords_t& c) const {
        //    if (i < c.i) return true;
        //    if (i > c.i) return false;
        //    if (j < c.j) return true;
        //    if (j > c.j) return false;
        //    if (t < c.t) return true;
        //    return false;
        //}

        bool operator ==(const coords_t& c) const {
            return i == c.i && j == c.j && t == c.t;
        }

        coords_t& operator +=(const coords_t& c) {
            i += c.i;
            j += c.j;
            t += c.t;
            return *this;
        }

        coords_t operator /(int s) const {
            coords_t c;
            c.i = i/s;
            c.j = j/s;
            c.t = t/s;
            return c;
        }

        std::string str() const {
            char buffer[128];
            sprintf(buffer, "(%d,%d,%d)", i,j,t);
            return buffer;
        }
    };

    struct coords_hash {
        size_t operator()(const coords_t& c) const {
            return
                ((size_t) c.i) <<  0 ^
                ((size_t) c.j) << 16 ^
                ((size_t) c.t) << 32;
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
        s_users eids;

        void add(const coords_t& c, const s_users& ids) {
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
        std::unordered_map<coords_t, s_users, coords_hash> _data;
        s_users _empty;

        const s_users& operator[](const coords_t& loc) const {
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

    struct user_data_t {
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

    //struct time_data_t {
    //    // t-> [{id},...]
    //    std::unordered_map<int, std::vector<std::set<std::string>>> _data;
    //};

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

        /// users
        mutable v_users _users;

        /// (i,j,t) -> {id, ...}
        sparse_data_t _sdata;

        /// id -> [(i,j,t), ...]
        user_data_t _udata;

        /// t -> [{id},...]
        //time_data_t _tdata;

    public:
        DiscreteWorld() {
            this->side(100);
            this->interval(5);
        }

        DiscreteWorld(double side, int minutes) {
            this->side(side);
            this->interval(minutes);
        }

        ~DiscreteWorld();

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
        // Get parameters
        //

        double side() const { return _side; }
        const time_duration& interval() const { return _interval; }

        //
        // Populate
        //

        void add(const std::string& id, double latitude, double longitude, const ptime& timestamp);

        // ----------------------------------------------------------------------

        const v_users& ids() const;

        // id: map[t -> set[id]]
        //void get_encounters(std::map<int, encounters_set_t>& encs, const std::string& id) const;

        // t -> [{id},...]
        void get_time_encounters(std::map<int, vs_users>& encs) const;

        // ----------------------------------------------------------------------
        // Conversions

        coords_t to_coords(double latitude, double longitude, const ptime& timestamp);

        ptime to_timestamp(int t) const;

        dwpoint_t to_point(const coords_t& c) const;

        // ------------------------------------------------------------------
        // IO

        void save(const std::string filename) const;

        void load(const std::string filename);

        template<class Archive>
        void save(Archive & ar) const
        {
            int seconds = _interval.total_seconds();
            ar(_side, _angle, seconds, _sdata._data, _udata._data);
        }
        template<class Archive>
        void load(Archive & ar)
        {
            int seconds;
            ar(_side, _angle, seconds, _sdata._data, _udata._data);
            _interval = time_duration(0, 0, seconds);
        }


        // ----------------------------------------------------------------------

        void save_slot_encounters(const std::string filename);

        void save_time_encounters(const std::string filename);

        // ----------------------------------------------------------------------

        void dump();

    };

    // ----------------------------------------------------------------------

} } }

extern void create_grid(int side, int interval, const std::string& path);

extern void load_grid(hls::khalifa::summer::DiscreteWorld& dworld, const std::string& fname);

#endif //CHECK_TRACKS_DWORLD_H
