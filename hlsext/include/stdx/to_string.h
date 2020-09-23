//
// Created by Corrado Mio on 19/09/2020.
//

#ifndef CHECK_TRACKS_TO_STRING_H
#define CHECK_TRACKS_TO_STRING_H

#pragma once

#include <iosfwd>
#include <vector>
#include <set>
#include <unordered_set>

namespace stdx {

    template<typename _Tp>
    std::string str(const std::vector<_Tp>& v, const std::string sep) {
        std::stringstream sbuf;

        sbuf << "[";
        if (!v.empty()) {
            sbuf << v[0];
            for(size_t i=1; i<v.size(); ++i)
                sbuf << sep << v[i];
        }

        sbuf << "]";
        return sbuf.str();
    }

    template<typename _Tp>
    std::string str(const std::set<_Tp>& v, const std::string sep) {
        std::stringstream sbuf;

        sbuf << "{";
        if (!v.empty()) {
            auto it = v.begin();
            sbuf << (*it);
            for(it++; it != v.end(); it++)
                sbuf << sep << (*it);
        }

        sbuf << "}";
        return sbuf.str();
    }

    template<typename _Tp>
    std::string str(const std::unordered_set<_Tp>& v, const std::string sep) {
        std::stringstream sbuf;

        sbuf << "{";
        if (!v.empty()) {
            auto it = v.begin();
            sbuf << (*it);
            for(it++; it != v.end(); it++)
                sbuf << sep << (*it);
        }

        sbuf << "}";
        return sbuf.str();
    }

}

#endif //CHECK_TRACKS_TO_STRING_H
