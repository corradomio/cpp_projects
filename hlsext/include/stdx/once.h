//
// Created by Corrado Mio on 30/03/2023.
//

#ifndef CHECK_ASSIGNMENT_ONCE_H
#define CHECK_ASSIGNMENT_ONCE_H

#include "exception.h"

namespace stdx {

    struct already_assigned_exception : stdx::exception {};

    template<typename T>
    struct once {
        T val;
        bool assigned;

        once(): val(T()), assigned(false) {}
        explicit once(const T& v): val(v), assigned(true) {}

        const T& get() const { return val; }

        operator const T&() const{ return val; }

        const T& operator=(const T& v) {
            if (assigned)
                throw already_assigned_exception();
            val = v;
            assigned = true;
            return val;
        }

        T detach() {
            assigned = false;
            return val;
        }

        operator const T&() {
            return val;
        }
    };

}

#endif //CHECK_ASSIGNMENT_ONCE_H
