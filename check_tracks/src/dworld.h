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
#include <stdx/default_unordered_map.h>


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

        coords_t& operator =(const coords_t& c) {
            this->i = c.i;
            this->j = c.j;
            this->t = c.t;
            return *this;
        }

        bool operator ==(const coords_t& c) const {
            return i == c.i && j == c.j && t == c.t;
        }

        //coords_t& operator +=(const coords_t& c) {
        //    i += c.i;
        //    j += c.j;
        //    t += c.t;
        //    return *this;
        //}

        //coords_t operator /(int s) const {
        //    coords_t c;
        //    c.i = i/s;
        //    c.j = j/s;
        //    c.t = t/s;
        //    return c;
        //}

        std::string str() const {
            char buffer[128];
            sprintf(buffer, "%d,%d,%d", i,j,t);
            return buffer;
        }
    };


    template<class Archive>
    void serialize(Archive & ar, coords_t& c)
    {
        ar(c.i, c.j, c.t);
    }

}}}

namespace hks = hls::khalifa::summer;

namespace std {
    template<>
    struct hash<hks::coords_t> {
        size_t operator()(const hls::khalifa::summer::coords_t& c) const {
            return
                ((size_t) c.i) <<  0 ^
                ((size_t) c.j) << 16 ^
                ((size_t) c.t) << 32;
        }
    };
}


//typedef std::string                 user_t;     // an user
typedef int user_t;
typedef std::unordered_set<user_t> s_users;     // set of users
typedef std::vector<s_users>      vs_users;     // vector of sets of users
typedef stdx::default_unordered_map<hks::coords_t, s_users>    c_users;     // coords -> set of users
typedef std::unordered_map<user_t, std::vector<hks::coords_t>> u_coords;    // user -> list of coords

namespace hls {
namespace khalifa {
namespace summer {

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

        /// {user, ...}
        s_users _susers;

        /// content of each cell
        /// (i,j,t) -> {user, ...}
        c_users _cusers;

        /// user -> [(i,j,t), ...]
        u_coords _ucoords;

        /// t -> [{user, ...}, ...]
        std::map<int, vs_users> _encs;
        bool _merged;

        void time_encounters();
        void merge_encounters();

    public:
        DiscreteWorld();
        DiscreteWorld(int size, int minutes);
        ~DiscreteWorld();

        //
        // Parameters
        //

        /// Length of the cell side (in meters)
        DiscreteWorld& side(int side);
        int side() const { return _side; }

        /// length of the time interval (in minutes)
        DiscreteWorld& interval(int minutes);
        int interval() const { return (int)(_interval.total_seconds()/60); }

        /// length of the time interval in 'boost::posix_time::time_duration'
        const time_duration& interval_td() const { return _interval; }

        ///
        DiscreteWorld& merged(bool m) { this->_merged = m; return *this; }
        bool           merged() const { return _merged; }

        //
        // Populate
        //

        /// populate the discrete world
        void add(const user_t& user, double latitude, double longitude, const ptime& timestamp);

        /// finalize the data structure
        void done();

        // ----------------------------------------------------------------------
        // Extract information
        // ----------------------------------------------------------------------

        /// set of users presents in the world
        const s_users& users() const { return _susers; }
        s_users users(double quota) const { return users((int)(quota*_susers.size())); }
        s_users users(int n) const;

        const std::map<int, vs_users>& get_time_encounters() const { return _encs; }

        // ----------------------------------------------------------------------
        // Conversions

        /// convert a discrete world coordinate in standard format
        coords_t to_coords(double latitude, double longitude, const ptime& timestamp);

        /// convert a time slot in standard format
        ptime to_timestamp(int t) const;

        // ------------------------------------------------------------------
        // IO

        /// save in the file
        void save(const std::string& filename) const;

        /// load from file
        void load(const std::string& filename);

        /// save slot encounters
        void save_slot_encounters(const std::string& filename);

        /// save time encounters
        void save_time_encounters(const std::string& filename);

        // ------------------------------------------------------------------
        // Cereal's serialization support
        // ------------------------------------------------------------------

        template<class Archive>
        void save(Archive & ar) const
        {
            int seconds = _interval.total_seconds();
            ar(_side, _angle, seconds, _susers, _cusers, _ucoords, _merged, _encs);
        }

        template<class Archive>
        void load(Archive & ar)
        {
            int seconds;
            ar(_side, _angle, seconds, _susers, _cusers, _ucoords, _merged, _encs);
            _interval = time_duration(0, 0, seconds);
        }

        // ----------------------------------------------------------------------
        // Debug
        // ----------------------------------------------------------------------

        void dump();

    };

} } }

/**
 * Create DataWorld with the specified parameters, ppulate it and save it
 * in the specified file
 *
 * @param side      length of the cell side
 * @param interval  length of the time interval
 * @param path      path where to save DiscreteWorld
 */
extern void create_grid(int side, int interval, const std::string& path);

#endif //CHECK_TRACKS_DWORLD_H
