//
// Created by Corrado Mio on 28/02/2016.
//

#ifndef TBBTEST_RANGE_HPP
#define TBBTEST_RANGE_HPP

#include "task_defs.hpp"

namespace hls {
namespace task {

    template<typename Index>
    class range {
        typedef Index const_iterator;
        typedef std::size_t size_type;

        Index _begin, _end;
        size_type _grainsize;
    public:
        range(): _begin(), _end(), _grainsize(1) { }
        range(const Index end):
            _begin(0), _end(end), _grainsize(1)
            { }
        range(const Index begin, const Index end, const size_type gs=1):
            _begin(begin), _end(end), _grainsize(gs)
            { }

        const_iterator begin() const { return _begin; }
        const_iterator   end() const { return _end;   }

        size_type size() const { return _end - _begin; }
        size_type grainsize() const { return _grainsize; }

        bool empty() const { return !(_begin < _end); }
        bool is_divisible() const { return  grainsize() < size(); }

    public:
        range(range& r, split):
            _begin(r._begin),
            _end(do_split(r)),
            _grainsize(r._grainsize)
            { }

    private:
        static Index do_split(range& r) {
            Index middle = r._begin + (r._end - r._begin)/2u;
            return r._begin = middle;
        }
    };
}};

#endif //TBBTEST_RANGE_HPP
