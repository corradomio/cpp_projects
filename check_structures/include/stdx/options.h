//
// Created by Corrado Mio on 18/03/2024.
//
/*
 * The idea is to have a generic object where to save extra options
 * to pass to a function.
 * The functions also has default values for all options:
 *
 *      options = {
 *          'name': value,
 *          ...
 *      }
 *
 *      options.get('name', default_value)
 *      options.get<type>('name', default_options)
 *
 * The values can be passed as string or original type
 */

#ifndef STDX_OPTIONS_H
#define STDX_OPTIONS_H

#include "language.h"
#include <string>
#include <map>
#include <memory>


namespace stdx {

    /// Options for complex functions
    /// Each option has a name and a primitive type.
    /// Supported types are: bool, int, long, double, string
    ///
    class options_t {
    public:
        enum value_type { BOOL, LONG, DOUBLE, STRING };
        struct value_t;
    private:
        std::map<std::string, value_t*> opts;

        void clear();
        void assign(const options_t& opts);
    public:
        options_t(){ }
        options_t(const options_t& opts) { assign(opts); }
        ~options_t() { clear(); }

        // options_t& set(const std::string& name, bool value);
        // options_t& set(const std::string& name, int value);
        // options_t& set(const std::string& name, long value);
        // options_t& set(const std::string& name, double value);
        // options_t& set(const std::string& name, const std::string& value);

        template<typename T>
        options_t& set(const std::string& name, T value);

        // bool       get(const std::string& name, bool defvalue) const;
        // int        get(const std::string& name, int defvalue) const;
        // long       get(const std::string& name, long defvalue) const;
        // double     get(const std::string& name, double defvalue) const;
        // const std::string& get(const std::string& name, const std::string& defvalue) const;

        template<typename T>
        T get(const std::string& name, T defvalue) const;

        // bool        get_bool(  const std::string& name, const options_t& defaults) const;
        // int         get_int(const std::string& name, const options_t& defaults) const;
        // long        get_long(  const std::string& name, const options_t& defaults) const;
        // double      get_double(const std::string& name, const options_t& defaults) const;
        // const std::string& get_string(const std::string& name, const options_t& defaults) const;

        template<typename T>
        T get(const std::string& name, const options_t& defaults) const;

        options_t& operator =(const options_t& other) {
            clear();
            assign(other);
            return self;
        }

    };

}

#endif //STDX_OPTIONS_H
