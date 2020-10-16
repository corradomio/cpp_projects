//
// Created by Corrado Mio on 10/10/2020.
//

#include "properties_utils.h"

namespace hls {
namespace util {

    const std::string TRIM_DELIMITERS = " \f\n\r\t\v";

    static std::string ltrim(const std::string &str) {
        std::string::size_type s = str.find_first_not_of(TRIM_DELIMITERS);
        if (s == std::string::npos) {
            return "";
        }
        return str.substr(s);
    }

    std::string right_trim(const std::string &str) {
        std::string::size_type s = str.find_last_not_of(TRIM_DELIMITERS);
        if (s == std::string::npos) {
            return "";
        }
        return str.substr(0, s + 1);
    }

    std::string left_trim(const std::string &str) {
        std::string rstr = ltrim(str);

        while (rstr != ltrim(rstr)) {
            rstr = ltrim(rstr);
        }

        return rstr;
    }

    std::string trim(const std::string &str) {
        return right_trim(left_trim(str));
    }

    bool is_property(const std::string &str) {
        std::string trimmedStr = left_trim(str);
        std::string::size_type s = trimmedStr.find_first_of("=");
        if (s == std::string::npos) {
            return false;
        }
        std::string key = trim(trimmedStr.substr(0, s));
        // key can't be empty
        if (key == "") {
            return false;
        }
        return true;
    }

    std::pair<std::string, std::string> parse_property(const std::string &str) {
        std::string trimmedStr = left_trim(str);
        std::string::size_type s = trimmedStr.find_first_of("=");
        std::string key = trim(trimmedStr.substr(0, s));
        std::string value = left_trim(trimmedStr.substr(s + 1));

        return std::pair<std::string, std::string>(key, value);
    }

    bool is_comment(const std::string &str) {
        std::string trimmedStr = left_trim(str);
        return trimmedStr[0] == '#';
    }

    bool is_empty_line(const std::string &str) {
        std::string trimmedStr = left_trim(str);
        return trimmedStr == "";
    }

}}
