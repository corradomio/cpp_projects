//
// Created by Corrado Mio on 16/10/2020.
//
#include <vector>
#include <string>
#include <regex>
#include <algorithm>
#include "stdx/strings.h"


std::vector<std::string> stdx::split(const std::string& s, const std::string& re) {
    std::regex word_regex{re};
    std::vector<std::string> parts;
    auto words_begin =
        std::sregex_iterator(s.begin(), s.end(), word_regex);
    auto words_end =
        std::sregex_iterator();

    size_t at = 0;
    for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
        auto match = (*i);
        int pos = match.position();
        size_t len = match.length();
        if (pos == at) continue;

        std::string part = s.substr(at, pos);
        parts.push_back(part);

        at = pos+len;
    }

    return parts;
}


std::set<std::string> stdx::tokens(const std::string& s, const std::string& re) {
    std::regex word_regex{re};
    std::set<std::string> parts;
    auto words_begin =
        std::sregex_iterator(s.begin(), s.end(), word_regex);
    auto words_end = std::sregex_iterator();

    for (std::sregex_iterator i = words_begin; i != words_end; ++i)
        parts.insert((*i).str());

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

// std::string stdx::tolower(const std::string& str) {
//     std::string upd{str};
//     std::transform(str.begin(), str.end(), upd.begin(),
//                    [](unsigned char c){ return std::tolower(c); });
//     return upd;
// }

void stdx::tolower(std::string& str) {
    std::transform(str.begin(), str.end(), str.begin(),
                   [](unsigned char c){ return std::tolower(c); });
}