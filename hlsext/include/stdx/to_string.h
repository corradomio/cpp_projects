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

#include <map>
#include <unordered_map>


using namespace std;

namespace stdx {

    //template<typename _Tp> std::string to_string(const std::list<_Tp>& v);
    //template<typename _Tp> std::string to_string(const std::vector<_Tp>& v);
    //template<typename _Tp> std::string to_string(const std::set<_Tp>& v);
    //template<typename _Tp> std::string to_string(const std::unordered_set<_Tp>& v);
    //template<typename _Kp, typename _Tp> std::string to_string(const std::map<_Kp,_Tp>& v);
    //template<typename _Kp, typename _Tp> std::string to_string(const std::unordered_map<_Kp, _Tp>& v);

    //template<typename _Tp> std::string to_string(const std::list<_Tp>& v, const std::string& sep);
    //template<typename _Tp> std::string to_string(const std::vector<_Tp>& v, const std::string& sep);
    //template<typename _Tp> std::string to_string(const std::set<_Tp>& v, const std::string& sep);
    //template<typename _Tp> std::string to_string(const std::unordered_set<_Tp>& v, const std::string& sep);
    //template<typename _Kp, typename _Tp> std::string to_string(const std::map<_Kp, _Tp>& v, const std::string& sep);
    //template<typename _Kp, typename _Tp> std::string to_string(const std::unordered_map<_Kp, _Tp>& v, const std::string& sep);

    //template<typename _Kp, typename _Tp> std::string to_string(const std::pair<_Kp, _Tp>& v) {
    //    return std::to_string(v.first) + ":" + std::to_string(v.second);
    //}

    //template<typename _Tp>
    //std::string to_string(const _Tp& v) {
    //    return std::to_string(v);
    //}

    //template<>
    //inline std::string to_string<std::string>(const std::string& s) { return s; }

    //template<typename _Tp>
    //std::string to_string(const std::list<_Tp>& v) {
    //    std::string sep = ",";
    //    return stdx::to_string(v, sep);
    //}

    //template<typename _Tp>
    //std::string to_string(const std::vector<_Tp>& v) {
    //    std::string sep = ",";
    //    return stdx::to_string(v, sep);
    //}

    //template<typename _Tp>
    //std::string to_string(const std::set<_Tp>& s) {
    //    std::string sep = ",";
    //    return stdx::to_string(s, sep);
    //}

    //template<typename _Tp>
    //std::string to_string(const std::unordered_set<_Tp>& s) {
    //    std::string sep = ",";
    //    return stdx::to_string(s, sep);
    //}

    //template<typename _Kp, typename _Tp>
    //std::string to_string(const std::map<_Kp,_Tp>& s) {
    //    std::string sep = ",";
    //    return stdx::to_string(s, sep);
    //}

    //template<typename _Kp, typename _Tp>
    //std::string to_string(const std::unordered_map<_Kp,_Tp>& s) {
    //    std::string sep = ",";
    //    return stdx::to_string(s, sep);
    //}


    //template<typename _Tp>
    //std::string to_string(const std::list<_Tp>& v, const std::string& sep) {
    //    std::string sbuf;
    //
    //    sbuf.append("[");
    //    if (!v.empty()) {
    //        auto it = v.begin();
    //        sbuf.append(stdx::to_string(*it));
    //        for(it++; it != v.end(); it++)
    //            sbuf.append(sep).append(stdx::to_string(*it));
    //    }
    //
    //    sbuf.append("]");
    //    return sbuf;
    //}

    //template<typename _Tp>
    //std::string to_string(const std::vector<_Tp>& v, const std::string& sep) {
    //    std::string sbuf;
    //
    //    sbuf.append("[");
    //    if (!v.empty()) {
    //        sbuf.append(stdx::to_string(v[0]));
    //        for(size_t i=1; i<v.size(); ++i)
    //            sbuf.append(sep).append(stdx::to_string(v[i]));
    //    }
    //
    //    sbuf.append("]");
    //    return sbuf;
    //}

    //template<typename _Tp>
    //std::string to_string(const std::set<_Tp>& v, const std::string& sep) {
    //    std::string sbuf;
    //
    //    sbuf.append("{");
    //    if (!v.empty()) {
    //        auto it = v.begin();
    //        sbuf.append(stdx::to_string(*it));
    //        for(it++; it != v.end(); it++)
    //            sbuf.append(sep).append(stdx::to_string(*it));
    //    }
    //
    //    sbuf.append("}");
    //    return sbuf;
    //}

    //template<typename _Tp>
    //std::string to_string(const std::unordered_set<_Tp>& v, const std::string& sep) {
    //    std::string sbuf;
    //
    //    sbuf.append("{");
    //    if (!v.empty()) {
    //        auto it = v.begin();
    //        sbuf.append(stdx::to_string(*it));
    //        for(it++; it != v.end(); it++)
    //            sbuf.append(sep).append(stdx::to_string(*it));
    //    }
    //
    //    sbuf.append("}");
    //    return sbuf;
    //}


    //template<typename _Kp, typename _Tp>
    //std::string to_string(const std::map<_Kp, _Tp>& v, const std::string& sep) {
    //    std::string sbuf;
    //
    //    sbuf.append("{");
    //    if (!v.empty()) {
    //        auto it = v.begin();
    //        sbuf.append(stdx::to_string(*it));
    //        for(it++; it != v.end(); it++)
    //            sbuf.append(sep).append(stdx::to_string(*it));
    //    }
    //
    //    sbuf.append("}");
    //    return sbuf;
    //}

    //template<typename _Kp, typename _Tp>
    //std::string to_string(const std::unordered_map<_Kp, _Tp>& v, const std::string& sep) {
    //    std::string sbuf;
    //
    //    sbuf.append("{");
    //    if (!v.empty()) {
    //        auto it = v.begin();
    //        sbuf.append(stdx::to_string(*it));
    //        for(it++; it != v.end(); it++)
    //            sbuf.append(sep).append(stdx::to_string(*it));
    //    }
    //
    //    sbuf.append("}");
    //    return sbuf;
    //}

}

#endif //CHECK_TRACKS_TO_STRING_H

