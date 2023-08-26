//
// Created by Corrado Mio on 03/08/2023.
//

#ifndef CHECK_MF_LINALG_H
#define CHECK_MF_LINALG_H

/*
        scalar == tensor/rank == 0
        vector == tensor/rank == 1
        matrix == tensor/rank == 2

    Only C++23 permits to define

        operator[](size_t i1, size_t i2, ...)

    otherwise, it is possible to use only

        operator[](size_t i1)

    Porkaround: override operator()

        operator()(size_t i1, size_t i2, ...)

 */

#include <cstdint>

namespace hls::linalg {

    struct ref_t {
        mutable int ref;

        ref_t(): ref(1) { }
    };

    struct dim_t {
        int dim0;
        int dim1;

        dim_t(               ): dim0(-1), dim1(-1) { }
        explicit dim_t(int n ): dim0(n), dim1(-1) { }
        dim_t(int n, int m   ): dim0(n), dim1(m ) { }
        // dim_t(const dim_t& m): dim0(m.dim0),  dim1(m.dim1) { }
        dim_t(const dim_t& m) = default;

        dim_t& operator =(const dim_t& m) {
            dim0 = m.dim0;
            dim1 = m.dim1;
            return *this;
        }

        [[nodiscard]] int rank() const {
            if (dim0 == -1) return 0;
            if (dim1 == -1) return 1;
            return 2;
        }

        [[nodiscard]] size_t size() const {
            size_t sz = 1;
            if (dim0 != -1) sz *= dim0;
            if (dim1 != -1) sz *= dim1;
            return sz;
        }

        [[nodiscard]] size_t size(int dim) const {
            if (dim == 0) return dim0;
            if (dim == 1) return dim1;
            return -1;
        }

    };

    template<typename T>
    struct array {
        ref_t*  pref;
        T*      data;
        dim_t   meta;

        array(                     ): pref(new ref_t), meta(    ), data(nullptr) {}
        explicit array(int n       ): pref(new ref_t), meta(n   ), data(new T[n]) { }
        array(int n, T* data       ): pref(new ref_t), meta(n   ), data(data) { }
        array(int n, int m         ): pref(new ref_t), meta(n, m), data(new T[n * m]) { }
        array(int n, int m, T* data): pref(new ref_t), meta(n, m), data(data) { }

        // ------------------------------------------------------------------

        array(const array& a): pref(a.pref), meta(a.meta), data(a.data) { pref->ref++; }

        ~array() {
            if (0 == --pref->ref) {
                delete[] data;
                delete pref;
            }
        }

        array& operator =(const array& a) {
            a.pref->ref++;
            if (0 == --pref->ref) {
                delete[] data;
                delete pref;
            }
            data = a.data;
            pref = a.pref;
            meta = a.meta;
            return *this;
        }

        // ------------------------------------------------------------------

        [[nodiscard]] int    rank() const { return meta.rank(); }
        [[nodiscard]] size_t size() const { return meta.size(); }
        [[nodiscard]] size_t size(int dim) const { return meta.size(dim); }

        // ------------------------------------------------------------------

        T& operator[](int i) { return data[i]; }
        T& operator[](int i, int j) { return data[i*meta.dim1 + j]; }

    };


    struct range_t {
        //  for (i=0, i<len, ++i)
        //      k = start + i*step
        int start, len, step;
    };

    template<typename T>
    struct view_t {
        T* data;
        range_t a0;
        range_t a1;
    };

    view_t<float> view(array<float> ar, int i, int axis=0);

    array<float> create(int n);
    array<float> create(int n, int m);

}

#endif //CHECK_MF_LINALG_H
