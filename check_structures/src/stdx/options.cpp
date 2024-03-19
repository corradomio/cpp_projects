//
// Created by Corrado Mio on 18/03/2024.
//
#include "stdx/exceptions.h"
#include <cassert>
#include <cstring>
#include "stdx/options.h"

#define MAX_STR_LEN 16

namespace stdx {

    template<typename K, typename T>
    bool contains(const std::map<K,T>& map, const K& key) {
        return map.find(key) != map.end();
    }

    // ----------------------------------------------------------------------

    struct options_t::value_t {
        value_type type;

        value_t(value_type type): type(type) { }

        virtual bool   get_bool()   { throw stdx::not_implemented("get_bool"); }
        virtual long   get_long()   { throw stdx::not_implemented("get_long"); }
        virtual double get_double() { throw stdx::not_implemented("get_double"); }
        virtual const std::string& get_string() { throw stdx::not_implemented("get_string"); }
    };

    struct bool_value: public options_t::value_t {
        typedef options_t::value_t super;

        bool value;
        bool_value(bool value): super(options_t::value_type::BOOL), value(value) { }

        virtual bool get_bool() override { return self.value; }
    };

    struct long_value: public options_t::value_t {
        typedef options_t::value_t super;

        long value;
        long_value(long value): super(options_t::value_type::LONG), value(value) { }

        virtual long get_long() override { return self.value; }
    };

    struct double_value: public options_t::value_t {
        typedef options_t::value_t super;

        double value;
        double_value(double value): super(options_t::value_type::DOUBLE), value(value) { }

        virtual double get_double() override { return self.value; }
    };

    struct string_value: public options_t::value_t {
        typedef options_t::value_t super;

        const std::string& value;
        string_value(const std::string& value): super(options_t::value_type::STRING), value(value) { }

        virtual const std::string& get_string() override { return self.value; }
    };

    // ----------------------------------------------------------------------

    template<> options_t& options_t::set(const std::string& name, bool value) {
        auto* pv = new bool_value(value);
        self.opts[name] = pv;
        return self;
    }

    template<> options_t& options_t::set(const std::string& name, int value) {
        auto* pv = new long_value(value);
        self.opts[name] = pv;
        return self;
    }

    template<> options_t& options_t::set(const std::string& name, long value) {
        auto* pv = new long_value(value);
        self.opts[name] = pv;
        return self;
    }

    template<> options_t& options_t::set(const std::string& name, double value) {
        auto* pv = new double_value(value);
        self.opts[name] = pv;
        return self;
    }

    template<> options_t& options_t::set(const std::string& name, std::string value) {
        auto* pv = new string_value(value);
        self.opts[name] = pv;
        return self;
    }

    template<> options_t& options_t::set(const std::string& name, const char* value) {
        auto* pv = new string_value(value);
        self.opts[name] = pv;
        return self;
    }

    // --

    template<> bool options_t::get(const std::string& name, bool defvalue) const {
        if (!contains<std::string, value_t*>(self.opts, name))
            return defvalue;
        return self.opts.at(name)->get_bool();
    }

    template<> int options_t::get(const std::string& name, int defvalue) const {
        if (!contains<std::string, value_t*>(self.opts, name))
            return defvalue;
        return self.opts.at(name)->get_long();
    }

    template<> long options_t::get(const std::string& name, long defvalue) const {
        if (!contains<std::string, value_t*>(self.opts, name))
            return defvalue;
        return self.opts.at(name)->get_long();
    }

    template<> size_t options_t::get<size_t>(const std::string& name, size_t defvalue) const {
        if (!contains<std::string, value_t*>(self.opts, name))
            return defvalue;
        return self.opts.at(name)->get_long();
    }

    template<> double options_t::get(const std::string& name, double defvalue) const {
        if (!contains<std::string, value_t*>(self.opts, name))
            return defvalue;
        return self.opts.at(name)->get_double();
    }

    template<> std::string options_t::get(const std::string& name, std::string defvalue) const {
        if (!contains<std::string, value_t*>(self.opts, name))
            return defvalue;
        return self.opts.at(name)->get_string();
    }

    // ----------------------------------------------------------------------

    template<> bool options_t::get<bool>(const std::string& name, const options_t& defaults) const {
        bool defval = defaults.get(name, false);
        return self.get(name, defval);
    }

    template<> int options_t::get<int>(const std::string& name, const options_t& defaults) const {
        int defval = defaults.get(name, 0);
        return self.get(name, defval);
    }

    template<> long options_t::get<long>(const std::string& name, const options_t& defaults) const {
        long defval = defaults.get(name, 0L);
        return self.get(name, defval);
    }

    template<> size_t options_t::get<size_t>(const std::string& name, const options_t& defaults) const {
        long defval = defaults.get(name, 0L);
        return self.get(name, defval);
    }

    template<> double options_t::get<double>(const std::string& name, const options_t& defaults) const {
        double defval = defaults.get(name, 0.);
        return self.get(name, defval);
    }

    template<> std::string options_t::get<std::string>(const std::string& name, const options_t& defaults) const {
        std::string defval = defaults.get(name, std::string());
        return self.get(name, defval);
    }

    // ----------------------------------------------------------------------

    void options_t::clear() {
        for (auto it = self.opts.begin(); it != self.opts.end(); ++it) {
            value_t* pv = it->second;
            delete pv;
        }
        self.opts.clear();
    }

    void options_t::assign(const stdx::options_t &other) {
        for (auto it = other.opts.begin(); it != other.opts.end(); ++it) {
            switch(it->second->type) {
                case options_t::value_type::BOOL:
                    self.set(it->first, other.get(it->first, false));
                    break;
                case options_t::value_type::LONG:
                    self.set(it->first, other.get(it->first, 0L));
                    break;
                case options_t::value_type::DOUBLE:
                    self.set(it->first, other.get(it->first, 0.));
                    break;
                case options_t::value_type::STRING:
                    self.set(it->first, other.get(it->first, std::string()));
                    break;
                default:
                    throw stdx::not_implemented("option type");
            }
        }
    }

}