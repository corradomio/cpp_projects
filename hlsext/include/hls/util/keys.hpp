//
// Created by Corrado Mio on 17/10/2015.
//

#ifndef HLS_KEYS_HPP
#define HLS_KEYS_HPP

#include <map>


namespace hls {
namespace util {

    /***
     * Iterate on the keys of a std::map.
     *
     * Usage:
     *
     *       #include <map>
     *       #include "hls/iterable/iterable.hpp"
     *
     *       int main() {
     *           std::map<std::string, int> m{{"one",1},{"two",2}, {"three",3}};
     *
     *           for (auto k : keys(m)) {
     *               std::cout << k << std::endl;
     *           }
     *
     *           return 0;
     *       }
     *
     *
     */
    template<typename Map>
    class keys_t {
        const Map& map;

        class keys_it {
            typename Map::const_iterator it;
        public:
            keys_it(const typename Map::const_iterator& __it): it(__it){ }
            keys_it(const keys_it& that): it(that.it){ }

            keys_it& operator++() { ++it; return *this; }
            bool  operator !=(const keys_it& that) const { return it != that.it; }
            const typename Map::key_type& operator*() const { return (*it).first; }
        };
    public:
        keys_t(const Map& __map): map(__map){ }
        keys_t(const keys_t& that): map(that.map){ }

        keys_it begin() const { return keys_it(map.begin()); }
        keys_it   end() const { return keys_it(map.end()); }

        size_t size() const { return map.size();  }
        bool  empty() const { return map.empty(); }
    };

    template<typename Map>
    keys_t<Map> keys(const Map& __map){ return keys_t<Map>(__map); };


}}

#endif // HLS_COLLECTION_ITERABLE_HPP
