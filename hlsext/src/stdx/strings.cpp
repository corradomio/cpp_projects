//
// Created by Corrado Mio on 16/10/2020.
//

#include "stdx/strings.h"

std::vector<std::string> stdx::split(const std::string& str, const std::string& delim) {
    std::vector<std::string> parts;
    size_t start = 0;
    size_t end = str.find(delim, start);
    while (end != std::string::npos) {
        parts.push_back(str.substr(start, end));
        start = end + delim.length();
    }
    if (start < str.length())
        parts.push_back(str.substr(start));
    return parts;
}