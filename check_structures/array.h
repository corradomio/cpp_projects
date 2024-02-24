//
// Created by Corrado Mio on 11/02/2024.
//

#ifndef CHECK_STRUCTURES_ARRAY_H
#define CHECK_STRUCTURES_ARRAY_H

#include <cstddef>

namespace stdx {
    /**
     * Efficient implementation of a C-like array with only two main operations
     * 1) append a new element to the array
     * 2) access to the i-th element
     * @tparam T
     */

    template<typename T>
    struct array_t {
        size_t n;   // n of element
        size_t c;   // array capacity (max number of elements)
        T*  data;   // array containing the elements

        /// Constructor with capacity 0
        explicit array_t(): c(0), n(0), data(new T[0]) { }
        /// Constructor with specified capacity and specified number of elements
        explicit array_t(int c, int n=-1): c(c), n(n<0?c:n) {
            data = new T[c];
        }
        /// Destructor
        ~array_t() { delete[] data; }

              T& operator [](size_t i)       { return data[i]; }
        const T& operator [](size_t i) const { return data[i]; }

        [[nodiscard]] inline bool  empty() const { return n == 0; }
        [[nodiscard]] inline size_t size() const { return n; }
        [[nodiscard]] inline size_t capacity() const { return c; }

        /// Change the capacity.
        void resize(size_t c_) {
            if (c_ == c)
                return;

            if (c_< n) n = c_;
            T* ndata = new T[c_];
            for (int i=0; i<n; ++i)
                ndata[i] = data[i];
            delete[] data;
            c = c_;
            data = ndata;
        }

        /// Change the capacity and the number of available elements
        void allocate(size_t n_) {
            resize(n_);
            n = n_;
        }

        /// Add a new element at the end and return its position
        size_t add(const T& e) {
            size_t id = n;
            if (n >= c)
                resize(c == 0 ? 2 : 2*c);
            data[n++] = e;
            return id;
        }

        void clear() {
            delete[] data;
            n = 0;
            c = 0;
            data = new T[0];
        }
    };
}

#endif //CHECK_STRUCTURES_ARRAY_H
