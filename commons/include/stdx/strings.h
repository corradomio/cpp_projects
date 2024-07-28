//
// Created by Corrado Mio on 16/10/2020.
//

#ifndef STDX_STRINGS_H
#define STDX_STRINGS_H

#include <vector>
#include <set>
#include <string>
#include <algorithm>

namespace stdx {

    std::vector<std::string> split(const std::string& str, const std::string& re);
    std::set<std::string>    tokens(const std::string& str, const std::string& re);

    std::string replace(const std::string& str, const std::string& from, const std::string& to);

    // std::string tolower(const std::string& str);
    void tolower(std::string& str);
}

#endif //STDX_STRINGS_H
