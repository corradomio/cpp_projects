//
// Created by Corrado Mio on 16/12/2023.
//

//  s*T, T*s
//  T1+T2, T1-T2, T1*T2, T1/T2      element-wise
//  T1.T2

#ifndef LINALG_H
#define LINALG_H

#include <initializer_list>
#include "refc.h"

namespace linalg {

    // byte, short, int, long, float, double,
    // bool
    // uint8, uint16, uint32, uint64
    // int8, int16, int32, int64
    // float16, float32, float64
    // complex32, complex64, complex128

    // enum tensor_type {
    //     byte,
    //     int8, int16, int32, int64,
    //     float16, float32, float64,
    //     complex32, complex64, complex126,
    //     float_t=float32,
    //     double_t=float64
    // };

    struct dim_t {
        size_t rank;
        size_t avail;
        size_t* dim;

        dim_t();
        dim_t(const dim_t& d);
        dim_t(std::initializer_list<size_t> dims);
        ~dim_t();

        size_t size() const;
        dim_t& operator=(const dim_t& d);
    };

    struct tensor_t: public stdx::refcount_t {
        dim_t  dims;
        float* data;

        tensor_t();
        tensor_t(const std::initializer_list<size_t>& dims);
        tensor_t(const dim_t& dims);

        ~tensor_t();
    };

    tensor_t*  newt(const std::initializer_list<size_t>& dims);
    tensor_t* zeros(const std::initializer_list<size_t>& dims);
    tensor_t*  ones(const std::initializer_list<size_t>& dims);
    tensor_t*  rand(const std::initializer_list<size_t>& dims);
    tensor_t*  newt(tensor_t* t);
    tensor_t*  newt(float f);

    struct tensor : public stdx::refp<tensor_t> {
        tensor(): refp() { }
        tensor(tensor_t* pt): refp(pt) { }
        tensor(const tensor& t): refp(t) { }

        [[nodiscard]] tensor_t* ptr() const { return  get(); }
        [[nodiscard]] tensor_t& ref() const { return *get(); }

        float& operator[](size_t i)          { return at(i);    }
        float& operator[](size_t i, size_t j){ return at(i, j); }

        float& at() { return get()->data[0]; }
        float& at(size_t i) { return get()->data[i]; }
        float& at(size_t i, size_t j) { return get()->data[i*get()->dims.dim[1]+j]; }

        size_t rank() const { return get()->dims.rank; }
        size_t dim(size_t i) const { return get()->dims.dim[i]; }
        size_t size() const { return get()->dims.size(); }

        // tensor dot(tensor t);
    };
}

#endif //LINALG_H
