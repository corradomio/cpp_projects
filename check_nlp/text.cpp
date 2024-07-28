//
// Created by Corrado Mio on 28/07/2024.
//
#include <fstream>
#include <regex>
#include <cctype>
#include <language.h>
#include <stdx/strings.h>
#include "text.h"

void text_t::load(const std::string& path) {
    std::string line;
    std::ifstream is(path);
    while(std::getline(is, line)) {
        self._content.append(" ");
        self._content.append(line);
    }
}

void text_t::parse(const std::string &re, bool lower) {
    std::string s = self._content;
    std::regex word_regex{re};
    std::vector<std::string> parts;
    auto words_begin =
        std::sregex_iterator(s.begin(), s.end(), word_regex);
    auto words_end =
        std::sregex_iterator();

    for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
        std::string term = i->str();
        if (lower)
            stdx::tolower(term);
        self._terms.push_back(term);
        self._tokens.insert(term);
    }

}
