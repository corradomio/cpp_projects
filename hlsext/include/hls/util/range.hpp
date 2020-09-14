//
// Created by Corrado Mio on 18/10/2015.
//

#ifndef HLS_UTIL_RANGE_HPP
#define HLS_UTIL_RANGE_HPP

namespace hls {
namespace util {


    template<typename T, int S=1>
    class range_t {
        T _begin;
        T _end;
    public:

        class range_it {
            T it;
        public:
            range_it(T __it): it(__it) { }
            range_it(const range_it& it): it(it.it) { }

            range_it& operator++() { it+=S; return *this; }
            bool  operator !=(const range_it& that) const { return it < that.it; }
            T operator*() const { return it; }
        };

    public:
        range_t(T end): _begin(0), _end(end) { }
        range_t(T begin, T end): _begin(begin), _end(end) { }

        range_it begin() const { return range_it(_begin); }
        range_it   end() const { return range_it(_end); }

        size_t size() const { return (_end-_begin)/S;  }
        bool  empty() const { return _begin >= _end; }
    };


}}

#endif //TEST_RANGE_HPP
