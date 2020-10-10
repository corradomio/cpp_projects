//
// Created by Corrado Mio on 07/10/2020.
//

#include <fstream>
#include <algorithm>
#include "hls/util/properties.h"
#include "properties_utils.h"


namespace hls{
namespace util {


    properties::properties() {
    }

    properties::~properties() {
        _names.clear();
        _properties.clear();
    }

    bool properties::contains(const std::string &name) const {
        return !(_properties.find(name) == _properties.end());
    }

    std::string properties::get(const std::string& name) const {
        if (!contains(name)) {
            throw property_not_found_exception(name + " does not exist");
        }
        return _properties.at(name);
    }

    std::string properties::get(const std::string& name, const std::string& defaultValue) const {
        if (!contains(name)) {
            return defaultValue;
        }
        return _properties.at(name);
    }

    const std::vector<std::string>& properties::names() const {
        return _names;
    }

    void properties::insert(const std::string& name, const std::string& value) {
        if (!contains(name)) {
            _names.push_back(name);
        }
        _properties[name] = value;
    }

    void properties::erase(const std::string& name) {
        if (!contains(name)) {
            //throw property_not_found_exception(name + " does not exist");
            return;
        }
        _names.erase(std::remove(_names.begin(), _names.end(), name), _names.end());
        _properties.erase(name);
    }

    // ---------------------------------------

    bool properties::get(const std::string &name, bool defaultValue) const {
        if (!contains(name))
            return defaultValue;
        std::string value = get(name);
        if (value == "true" || value=="1" || value == "on" || value == "yes")
            return true;
        if (value == "false" || value=="0" || value == "off" || value == "no")
            return false;
        else
            return defaultValue;
    }

    long properties::get(const std::string &name, long defaultValue) const {
        if (!contains(name))
            return defaultValue;
        else
            return ::atol(get(name).c_str());
    }

    double properties::get(const std::string &name, double defaultValue) const {
        if (!contains(name))
            return defaultValue;
        else
            return ::atof(get(name).c_str());
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
            const std::vector<std::string>& names = props.names();
            for (std::vector<std::string>::const_iterator i = names.begin(); i != names.end(); ++i) {
                os << *i << " = " << props.get(*i) << std::endl;
            }
            os.close();
        } catch (...) {
            // don't forget to close the ofstream
            os.close();
            throw;
        }
    }

}}
