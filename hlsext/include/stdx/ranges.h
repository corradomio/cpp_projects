//
// Created by Corrado Mio on 23/09/2020.
//

#ifndef CHECK_HLSEXT_RANGES_H
#define CHECK_HLSEXT_RANGES_H

#include <iterator>
#include <vector>

namespace stdx {

    template<typename T>
    class range_t {
        T _begin;
        T _end;

    public:
        class iter_t {
            //const range_t& r;
            mutable T value;
        public:
            iter_t(const range_t& rng, const T& v): /*r(rng),*/ value(v) { }
            bool operator !=(const iter_t& it) { return value != it.value; }
            bool operator  <(const iter_t& it) { return value  < it.value; }

            iter_t& operator ++() {
                value += 1;
                return *this;
            }

            iter_t& operator ++(int) {
                value += 1;
                return *this;
            }

            T operator*() const { return value; }
        };

        friend class range_t::iter_t;

    public:
        typedef const iter_t const_iterator;

        range_t(const T& begin, const T& end): _begin(begin),_end(end) { }
        const_iterator begin() const { return iter_t(*this, _begin); }
        const_iterator   end() const { return iter_t(*this, _end); }
    };

    template<typename T> range_t<T> range(T end) { return range_t<T>(0, end); }
    template<typename T> range_t<T> range(T begin, T end) { return range_t<T>(begin, end); }

}

#endif //CHECK_HLSEXT_RANGES_H
