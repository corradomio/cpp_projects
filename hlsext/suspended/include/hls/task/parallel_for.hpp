//
// Created by Corrado Mio on 28/02/2016.
//

#ifndef TBBTEST_PARALLEL_FOR_HPP
#define TBBTEST_PARALLEL_FOR_HPP

#include "task_defs.hpp"
#include "partitioner.hpp"

namespace hls {
namespace task {

    namespace internal {

    };


    template<typename Range, typename Body, typename Partitioner>
    void parallel_for(const Range& range, const Body& body, const Partitioner& partitioner) {

    };

    template<typename Range, typename Body>
    void parallel_for(const Range& range, const Body& body) {
        parallel_for(range, body, auto_partitioner());
    };

}};

#endif //TBBTEST_PARALLEL_FOR_HPP
