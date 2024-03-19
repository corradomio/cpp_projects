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
    /// Supported types:
    ///
    ///     bool, int, long, double, string, size_t
    ///
    /// Initialization:
    ///
    ///     options_t().set(name1, value1).set(name2, vale2). ...
    ///
    /// or
    ///
    ///     options_t()(name1, value1)(name2, vale2) ...
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
        options_t() = default;
        options_t(const options_t& opts) { assign(opts); }
        ~options_t() { clear(); }

        template<typename T>
        options_t& set(const std::string& name, T value);

        template<typename T>
        T get(const std::string& name, T defvalue) const;

        template<typename T>
        T get(const std::string& name, const options_t& defaults) const;

        template<typename T>
        options_t& operator()(const std::string& name, T value) {
            set(name, value);
            return self;
        }

        options_t& operator =(const options_t& other) {
            clear();
            assign(other);
            return self;
        }

    };

}

#endif //STDX_OPTIONS_H
