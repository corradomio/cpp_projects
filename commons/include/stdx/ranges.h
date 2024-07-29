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
        T step;

    public:
        class iter_t {
            const range_t* r;
            mutable T value;
        public:
            typedef iter_t self_type;
            typedef T value_type;
            typedef T& reference;
            typedef T* pointer;
            typedef std::forward_iterator_tag iterator_category;
            typedef int difference_type;

        public:
            iter_t(const range_t<T>* r, T n): r(r), value(n) { }
            bool operator !=(const iter_t& it) const { return value != it.value; }
            bool operator  <(const iter_t& it) const { return value  < it.value; }

            iter_t operator ++()    { value += (*r).step; return self; }
            iter_t operator ++(int) { value += (*r).step; return self; }

            value_type operator*() const { return value; }
        };

        friend class range_t::iter_t;

    public:
        typedef const iter_t const_iterator;

        range_t(const T& begin, const T& end, const T& step): _begin(begin),_end(end),step(step) { }
        const_iterator  begin() const { return iter_t(this, _begin); }
        const_iterator    end() const { return iter_t(this, _end);   }
        // const_iterator cbegin() const { return iter_t(this, _begin); }
        // const_iterator   cend() const { return iter_t(this, _end);   }

        size_t size() const { return (_end - _begin)/step; }
    };

    template<typename T> range_t<T> range(const T& end)
        { return range_t<T>(0, end, 1); }
    template<typename T> range_t<T> range(const T& begin, const T& end)
        { return range_t<T>(begin, end, 1); }
    template<typename T> range_t<T> range(const T& begin, const T& end, const T& step)
        { return range_t<T>(begin, end, step); }

}

#endif //CHECK_HLSEXT_RANGES_H
