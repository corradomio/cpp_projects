//
// Created by Corrado Mio on 10/10/2020.
//

#ifndef HLSEXT_PROPERTIES_UTILS_H
#define HLSEXT_PROPERTIES_UTILS_H

#include <string>

namespace hls {
namespace util {

/**
 * Left trims a string.
 * This function doesn't modify the given str.
 */
    extern std::string left_trim(const std::string& str);

/**
 * Right trims a string.
 * This function doesn't modify the given str.
 */
    extern std::string right_trim(const std::string& str);

/**
 * Trims a string (perform a left and right trims).
 * This function doesn't modify the given str.
 */
    extern std::string trim(const std::string& str);

/**
 * Is a given string a property. A property must have the following format:
 * key=value
 */
    extern bool is_property(const std::string& str);

/**
 * Parses a given property into a pair of key and value.
 *
 * ParseProperty assumes a given string has a correct format.
 */
    extern std::pair<std::string, std::string> parse_property(const std::string& str);

/**
 * Is a given string a comment? A comment starts with #
 */
    extern bool is_comment(const std::string& str);

/**
 * Is a given string empty?
 */
    extern bool is_empty_line(const std::string& str);
}
}

#endif //HLSEXT_PROPERTIES_UTILS_H
