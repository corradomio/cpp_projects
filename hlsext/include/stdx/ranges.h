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
            typedef iter_t self_type;
            typedef T value_type;
            typedef T& reference;
            typedef T* pointer;
            typedef std::forward_iterator_tag iterator_category;
            typedef int difference_type;

        public:
            iter_t(const self_type& rng, const T& v): value(v) { }
            iter_t(const stdx::range_t<T>& rng, const T& n): value(rng._begin+n) { }
            bool operator !=(const self_type& it) const { return value != it.value; }
            bool operator  <(const self_type& it) const { return value  < it.value; }

            self_type operator ++() { self_type it = *this; value += 1; return it; }
            self_type operator ++(int) { value += 1; return *this; }

            value_type operator*() const { return value; }
        };

        friend class range_t::iter_t;

    public:
        typedef const iter_t const_iterator;

        range_t(const T& begin, const T& end): _begin(begin),_end(end) { }
        const_iterator  begin() { return iter_t(*this, _begin); }
        const_iterator    end() { return iter_t(*this, _end); }
        const_iterator cbegin() const { return iter_t(*this, _begin); }
        const_iterator   cend() const { return iter_t(*this, _end); }

        size_t size() const { return _end - _begin; }
    };

    template<typename T> range_t<T> range(T end) { return range_t<T>(0, end); }
    template<typename T> range_t<T> range(T begin, T end) { return range_t<T>(begin, end); }

}

#endif //CHECK_HLSEXT_RANGES_H
