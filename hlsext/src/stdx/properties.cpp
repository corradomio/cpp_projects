//
// Created by Corrado Mio on 07/10/2020.
//

#include <fstream>
#include <algorithm>
#include "stdx/properties.h"
#include "stdx/strings.h"
#include "../hls/util/properties_utils.h"


namespace stdx {


    properties::properties() {
    }

    properties::properties(const std::string& file) {
        read(*this, file);
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

    int properties::get(const std::string &name, int defaultValue) const {
        if (!contains(name))
            return defaultValue;
        else
            return ::atoi(get(name).c_str());
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

    int properties::get(const std::string& name, const std::vector<std::string>& enums) const {
        std::string value = get(name);
        for(int i=0; i<enums.size(); ++i)
            if (enums[i] == value)
                return i;
        return 0;
    }

    int properties::get(const std::string &name, const std::initializer_list<std::string>& init) const {
        std::vector<std::string> enums(init);
        std::string value = get(name);
        for(int i=0; i<enums.size(); ++i)
            if (enums[i] == value)
                return i;
        return 0;
    }

    // ----------------------------------------------------------------------

    std::vector<long> properties::get_longs(const std::string &name, const std::string &sep) const {
        std::string vlist = get(name, std::string(""));
        std::vector<std::string> parts = stdx::split(vlist, sep);
        std::vector<long> values;
        for(const std::string& v : parts) {
            long value = ::atol(v.c_str());
            values.push_back(value);
        }
        return values;
    }

    std::vector<int> properties::get_ints(const std::string &name, const std::string &sep) const {
        std::string vlist = get(name, std::string(""));
        std::vector<std::string> parts = stdx::split(vlist, sep);
        std::vector<int> values;
        for(const std::string& v : parts) {
            int value = ::atoi(v.c_str());
            values.push_back(value);
        }
        return values;
    }

    // ----------------------------------------------------------------------

    void properties::read(properties& props, const std::string& file) {

        std::ifstream is;
        is.open(file.c_str());
        if (!is.is_open()) {
            throw properties_exception("PropertiesParser::Read(" + file + "): Unable to open file for reading.");
        }

        try {
            size_t linenr = 0;
            std::string line;
            while (getline(is, line)) {
                if (hls::util::is_empty_line(line) || hls::util::is_comment(line)) {
                    // ignore it
                } else if (hls::util::is_property(line)) {
                    std::pair<std::string, std::string> prop = hls::util::parse_property(line);
                    props.insert(prop.first, prop.second);
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

}
