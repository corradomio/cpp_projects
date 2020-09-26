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

using namespace std;

namespace stdx {

    template<typename _Tp> std::string str(const std::vector<_Tp>& v);
    template<typename _Tp> std::string str(const std::set<_Tp>& v);
    template<typename _Tp> std::string str(const std::unordered_set<_Tp>& v);

    template<typename _Tp> std::string str(const std::vector<_Tp>& v, std::string& sep);
    template<typename _Tp> std::string str(const std::set<_Tp>& v, std::string& sep);
    template<typename _Tp> std::string str(const std::unordered_set<_Tp>& v, std::string& sep);

    template<typename _Tp>
    std::string str(const _Tp& v) {
        return std::to_string(v);
    }

    //template<typename _Tp>
    //std::string str(const _Tp& v, const char* chr) {
    //    std::string sep = chr;
    //    return stdx::str(v, sep);
    //}

    template<>
    inline std::string str<std::string>(const std::string& s) { return s; }

    template<typename _Tp>
    std::string str(const std::vector<_Tp>& v) {
        std::string sep = ";";
        return stdx::str(v, sep);
    }

    template<typename _Tp>
    std::string str(const std::set<_Tp>& s) {
        std::string sep = ";";
        return stdx::str(s, sep);
    }

    template<typename _Tp>
    std::string str(const std::unordered_set<_Tp>& s) {
        std::string sep = ";";
        return stdx::str(s, sep);
    }


    template<typename _Tp>
    std::string str(const std::vector<_Tp>& v, std::string& sep) {
        //std::stringstream sbuf;
        //
        //sbuf << "[";
        //if (!v.empty()) {
        //    sbuf << v[0];
        //    for(size_t i=1; i<v.size(); ++i)
        //        sbuf << sep << v[i];
        //}
        //
        //sbuf << "]";
        //return sbuf.str();

        std::string sbuf;

        sbuf.append("[");
        if (!v.empty()) {
            sbuf.append(stdx::str(v[0]));
            for(size_t i=1; i<v.size(); ++i)
                sbuf.append(sep).append(stdx::str(v[i]));
        }

        sbuf.append("]");
        return sbuf;
    }

    template<typename _Tp>
    std::string str(const std::set<_Tp>& v, std::string& sep) {
        //std::stringstream sbuf;
        //
        //sbuf << "{";
        //if (!v.empty()) {
        //    auto it = v.begin();
        //    sbuf << (*it);
        //    for(it++; it != v.end(); it++)
        //        sbuf << sep << (*it);
        //}
        //
        //sbuf << "}";
        //return sbuf.str();

        std::string sbuf;

        sbuf.append("{");
        if (!v.empty()) {
            auto it = v.begin();
            sbuf.append(stdx::str(*it));
            for(it++; it != v.end(); it++)
                sbuf.append(sep).append(stdx::str(*it));
        }

        sbuf.append("}");
        return sbuf;
    }

    template<typename _Tp>
    std::string str(const std::unordered_set<_Tp>& v, std::string& sep) {
        //std::stringstream sbuf;
        ////std::stringbuf sbuf;
        //
        //sbuf << "{";
        //if (!v.empty()) {
        //    auto it = v.begin();
        //    sbuf << (*it);
        //    for(it++; it != v.end(); it++)
        //        sbuf << sep << (*it);
        //}
        //
        //sbuf << "}";
        //return sbuf.str();

        std::string sbuf;

        sbuf.append("{");
        if (!v.empty()) {
            auto it = v.begin();
            sbuf.append(stdx::str(*it));
            for(it++; it != v.end(); it++)
                sbuf.append(sep).append(stdx::str(*it));
        }

        sbuf.append("}");
        return sbuf;
    }

}

#endif //CHECK_TRACKS_TO_STRING_H

