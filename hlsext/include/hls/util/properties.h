//
// Created by Corrado Mio on 07/10/2020.
//

#ifndef HLSEXT_PROPERTIES_H
#define HLSEXT_PROPERTIES_H

#include <string>
#include <vector>
#include <map>

namespace hls {
namespace util {

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


    /**
     * Classe simile a "Properties" di Java, utile per
     */
    class properties {
    public:
        properties();
        virtual ~properties();

        /**
         * Gets the property value from a given key.
         *
         * This method throws a PropertyNotFoundException when a given key does not
         * exist.
         */
        std::string at(const std::string& key) const;

        /**
         * Gets the property value from a given key. Use a default value if not found.
         */
        std::string at(const std::string& key, const std::string& defaultValue) const;

        /**
         * Gets the list of property names.
         */
        const std::vector<std::string>& propertyNames() const;

        /**
         * Adds a new property. If the property already exists, it'll overwrite
         * the old one.
         */
        void insert(const std::string& key, const std::string& value);

        /**
         * Removes the property from a given key.
         *
         * If the property doesn't exist a PropertyNotFoundException will be thrown.
         */
        void erase(const std::string& key);
    private:
        // to preserve the order
        std::vector<std::string> _names;
        std::map<std::string, std::string> _properties;

    public:

        /**
         * Reads a properties file and returns a Properties object.
         */
        static properties read(const std::string& file);

        /**
         * Writes Properties object to a file.
         */
        static void write(const std::string& file, const properties& props);
    };

}}

#endif //HLSEXT_PROPERTIES_H
