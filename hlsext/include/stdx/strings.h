//
// Created by Corrado Mio on 16/10/2020.
//

#ifndef HLSEXT_STRINGS_H
#define HLSEXT_STRINGS_H

#include <vector>
#include <string>
#include <algorithm>

namespace stdx {

    std::vector<std::string> split(const std::string& str, const std::string& delim);

    std::string replace(const std::string& str, const std::string& from, const std::string& to);

}

#endif //HLSEXT_STRINGS_H
