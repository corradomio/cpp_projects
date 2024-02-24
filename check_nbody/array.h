//
// Created by Corrado Mio on 15/02/2024.
//

#ifndef CHECK_NBODY_ARRAY_H
#define CHECK_NBODY_ARRAY_H

#include <cstddef>

namespace stdx {

    template<typename T>
    class array_t {
        size_t n;   // n of elements
        size_t c;   // capacity
        T* data;

    public:
        array_t(size_t c): n(0), c(c), data(new T[c]) { }
        array_t(size_t n, size_t c): n(n), c(c), data(new T[c]) { }
        ~array_t(){ delete[] data; }

        T& operator[](size_t i)       { return data[i]; }
        const T& operator[](size_t i) const { return data[i]; }

        size_t add(const T& e) {
            if (n == c) expand();
            data[n++] = e;
            return n-1;
        }

        [[nodiscard]] size_t size() const { return n; }

    private:
        void expand() {
            size_t nc = (c == 0) ? 2 : c*2;
            T* ndata = new T[nc];
            for (int i=0; i<n; ++i)
                ndata[i] = data[i];
            delete[] data;
            c = nc;
            data = ndata;
        }
    };

}

#endif //CHECK_NBODY_ARRAY_H
