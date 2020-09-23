//
// Created by Corrado Mio on 23/09/2020.
//

#ifndef CHECK_HLSEXT_RANGES_H
#define CHECK_HLSEXT_RANGES_H

#include <iterator>

namespace stdx {

    template<typename T>
    class range {
        T _begin;
        T _end;

    public:
        class iter {
            const range& r;
            T value;
        public:
            explicit iter(const range& rng, const T& v): r(rng), value(v) { }
            iter(const iter& it): r(it.r), value(it.value) { }

            iter& operator =(const iter& it) {
                value = it.value;
                return *this;
            }

            bool operator !=(const iter& it) { return value != it.value; }
            bool operator  <(const iter& it) { return value < it.value; }

            iter& operator ++() {
                value += 1;
                return *this;
            }

            iter& operator++(int) {
                value += 1;
                return *this;
            }

            T operator*() const { return value; }
        };

        friend class range::iter;

    public:
        range(const T& end): _begin(0),_end(end) { }
        range(const T& begin, const T& end): _begin(begin),_end(end) { }

        iter begin() const { return iter(*this, _begin); }
        iter   end() const { return iter(*this, _end); }

    };

}

#endif //CHECK_HLSEXT_RANGES_H
