//
// Created by Corrado Mio on 07/10/2020.
//

#include <fstream>
#include <algorithm>
#include "hls/util/properties.h"
#include "properties_utils.h"


using namespace hls::util;

properties::properties() {
}

hls::util::properties::~properties() {
    _names.clear();
    _properties.clear();
}

std::string properties::at(const std::string& key) const {
    if (_properties.find(key) == _properties.end()) {
        throw property_not_found_exception(key + " does not exist");
    }
    return _properties.at(key);
}

std::string properties::at(const std::string& key, const std::string& defaultValue) const {
    if (_properties.find(key) == _properties.end()) {
        return defaultValue;
    }
    return _properties.at(key);
}

const std::vector<std::string>& properties::propertyNames() const {
    return _names;
}

void properties::insert(const std::string& key, const std::string& value) {
    if (_properties.find(key) == _properties.end()) {
        _names.push_back(key);
    }
    _properties[key] = value;
}

void properties::erase(const std::string& key) {
    if (_properties.find(key) == _properties.end()) {
        throw property_not_found_exception(key + " does not exist");
    }
    _names.erase(std::remove(_names.begin(), _names.end(), key), _names.end());
    _properties.erase(key);
}

// ---------------------------------------


properties properties::read(const std::string& file) {
    properties properties;

    std::ifstream is;
    is.open(file.c_str());
    if (!is.is_open()) {
        throw properties_exception("PropertiesParser::Read(" + file + "): Unable to open file for reading.");
    }

    try {
        size_t linenr = 0;
        std::string line;
        while (getline(is, line)) {
            if (is_empty_line(line) || is_comment(line)) {
                // ignore it
            } else if (is_property(line)) {
                std::pair<std::string, std::string> prop = parse_property(line);
                properties.insert(prop.first, prop.second);
            } else {
                throw properties_exception("PropertiesParser::Read(" + file + "): Invalid line " + std::to_string(linenr) + ": " + line);
            }
            ++linenr;
        }
        is.close();
    } catch (...) {
        // don't forget to close the ifstream
        is.close();
        throw;
    }

    return properties;
}

void properties::write(const std::string& file, const properties& props) {
    std::ofstream os;
    os.open(file.c_str());
    if (!os.is_open()) {
        throw properties_exception("PropertiesParser::Write(" + file + "): Unable to open file for writing.");
    }

    try {
        const std::vector<std::string>& keys = props.propertyNames();
        for (std::vector<std::string>::const_iterator i = keys.begin(); i != keys.end(); ++i) {
            os << *i << " = " << props.at(*i) << std::endl;
        }
        os.close();
    } catch (...) {
        // don't forget to close the ofstream
        os.close();
        throw;
    }
}