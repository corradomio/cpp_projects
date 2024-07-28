//
// Created by Corrado Mio on 28/07/2024.
//

#ifndef CHECK_NLP_TEXT_H
#define CHECK_NLP_TEXT_H

#include <language.h>
#include <vector>
#include <set>
#include <string>
#include <stdx/bag.h>


class text_t {
    std::string _re;
    std::string _content;
    mutable std::vector<std::string> _terms;
    mutable stdx::bag<std::string> _tokens;
public:
    text_t(): _re("\\w") { }
    text_t(const std::string& path) {
        self.load(path);
    }

    void load(const std::string& path);

    void clear() {
        self._content.clear();
    }

    [[nodiscard]] size_t length()    const { return self._content.length(); }
    [[nodiscard]] std::string text() const { return self._content; }

    [[nodiscard]] stdx::bag<std::string>& tokens()  const { return self._tokens; }
    [[nodiscard]] std::vector<std::string>& terms() const { return self._terms;  }

    void parse(const std::string& re, bool lower=true);

};


#endif //CHECK_NLP_TEXT_H
