//
// Created by Corrado Mio on 05/08/2018.
//

#ifndef GENERIC_PRIMITIVE_HPP
#define GENERIC_PRIMITIVE_HPP

namespace stdx {

    template<typename T>
    class primitive_t {
        T value;
    public:
        primitive_t(): value(T()) { }
        primitive_t(const T v): value(v) { }
        primitive_t(const primitive_t& p): value(p.value) { }

        operator T() const { return value; }
        // operator const T() const { return value; }

        // primitive_t& operator =(const primitive_t& p) { value = p.value; return *this; }
        // primitive_t& operator =(T v) { value = v; return *this; }
    };

}

#endif //GENERIC_PRIMITIVE_HPP
