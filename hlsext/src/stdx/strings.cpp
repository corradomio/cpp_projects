//
// Created by Corrado Mio on 16/10/2020.
//

#include "stdx/strings.h"

std::vector<std::string> stdx::split(const std::string& str, const std::string& delim) {
    std::vector<std::string> parts;
    size_t start = 0;
    size_t end = str.find(delim, start);
    while (end != std::string::npos) {
        parts.push_back(str.substr(start, end-start));
        start = end + delim.length();
        end = str.find(delim, start);
    }
    if (start < str.length())
        parts.push_back(str.substr(start));
    return parts;
}

std::string stdx::replace(const std::string& cstr, const std::string& from, const std::string& to) {
    std::string str = cstr;
    size_t start_pos = str.find(from);
    while (start_pos != std::string::npos){
        str.replace(start_pos, from.length(), to);
        start_pos = str.find(from);
    }
    return str;
}