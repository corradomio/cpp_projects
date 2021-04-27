//
// Created by Corrado Mio (Local) on 27/04/2021.
//

#ifndef PARALLEL_H
#define PARALLEL_H

namespace stdx {

    template<typename T, typename Body>
    void parallel_for(const T& begin, const T& end, const Body& body) {
        for(auto index=begin; index != end; ++index)
            body(index);
    }

    template<typename Collection, typename Body>
    void parallel_foreach(const Collection& collection, const Body& body) {
        for(auto elem: collection)
            body(elem);
    }

    template<typename Iterator, typename Body>
    void parallel_foreach(const Iterator& begin, const Iterator& end, const Body& body) {
        for(auto it=begin; it != end; it++)
            body(*it);
    }
}

#endif //CHECK_PFOR_PARALLEL_H
