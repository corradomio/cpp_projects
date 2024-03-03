//
// Created by Corrado Mio on 29/02/2024.
//

#ifndef STDX_VECTOR_H
#define STDX_VECTOR_H

#include "array.h"

namespace stdx {

    template<typename T>
    struct vector_t : public array_t<T> {

        void assign(const vector_t &that) {
            array_t<T>::assign(that);
        }

        vector_t clone() const {
            return vector_t(self, true);
        }

        // ------------------------------------------------------------------

        vector_t() { }
        explicit vector_t(size_t n): array_t<T>(n, n) { }

        // implicit conversion
        vector_t(const array_t<T> &v): vector_t(v, false){ }

        // implicit conversion
        vector_t(const array_t<T> &v, bool clone): array_t<T>(v, clone){ }

        // ------------------------------------------------------------------

        vector_t &operator =(const array_t<T> &v) {
            assign(v);
            return self;
        }

        vector_t &operator =(T s) {
            array_t<T>::fill(s);
            return self;
        }

        // ------------------------------------------------------------------

    };

}

#endif //STDX_VECTOR_H
