//
// Created by Corrado Mio on 07/10/2020.
//

#ifndef HLSEXT_PROPERTIES_H
#define HLSEXT_PROPERTIES_H

#include <string>
#include <vector>
#include <map>

namespace stdx {

    class properties_exception : public std::exception {
    public:
        properties_exception() {}
        properties_exception(const std::string& msg) throw() : message(msg) {}

        virtual ~properties_exception() throw() {}

        const std::string& str() const throw() { return message; }

        virtual const char* what() const throw() { return message.c_str(); }

    private:
        std::string message;
    };

    class property_not_found_exception : public std::exception {
    public:
        property_not_found_exception() {}
        property_not_found_exception(const property_not_found_exception& e): message(e.message) {}
        property_not_found_exception(const std::string& msg) : message(msg) {}

        virtual ~property_not_found_exception() throw() {}

        const std::string& str() const throw() { return message; }

        virtual const char* what() const throw() { return message.c_str(); }

    private:
        std::string message;
    };


    class properties {
    public:
        properties();
        explicit properties(const std::string& file);
        virtual ~properties();

        /**
         * Check if the property is present
         */
        bool contains(const std::string& name) const;

        /**
         * Gets the property value from a given name.
         *
         * This method throws a PropertyNotFoundException when a given name does not
         * exist.
         */
        std::string get(const std::string& name) const;
        std::string get(const std::string& name, const std::string& defaultValue) const;

        // stl collections
        std::string at(const std::string& name) const { return get(name); }
        std::string operator[] (const std::string& name) const { return get(name); }

        // -- specialized
        bool get(const std::string& name, bool defaultValue) const;
        int get(const std::string& name, int defaultValue) const;
        long get(const std::string& name, long defaultValue) const;
        double get(const std::string& name, double defaultValue) const;

        std::vector<long> get_longs(const std::string& name, const std::string& sep=",") const;
        std::vector<int> get_ints(const std::string& name, const std::string& sep=",") const;

        /**
         * Gets the list of property names.
         */
        const std::vector<std::string>& names() const;

        /**
         * Adds a new property. If the property already exists, it'll overwrite
         * the old one.
         */
        void insert(const std::string& name, const std::string& value);

        /**
         * Removes the property from a given name.
         *
         * If the property doesn't exist a PropertyNotFoundException will be thrown.
         */
        void erase(const std::string& name);
    private:
        // to preserve the order
        std::vector<std::string> _names;
        std::map<std::string, std::string> _properties;

    public:

        /**
         * Reads a properties file and returns a Properties object.
         */
        static void read(properties& props, const std::string& file);

        /**
         * Writes Properties object to a file.
         */
        static void write(const std::string& file, const properties& props);
    };

}

#endif //HLSEXT_PROPERTIES_H
