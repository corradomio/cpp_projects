//
// Created by Corrado Mio on 07/03/2024.
//
#include "vector.h"

namespace stdx::float64 {

    void vector_t::init(const vector_t& that) {
        array_t::init(that);
    }

    void vector_t::assign(const vector_t& that) {
        array_t::assign(that);
    }

    void vector_t::fill(const vector_t& that) {
        array_t::fill(that);
    }

    void vector_t::fill(real_t s) {
        array_t::fill(s);
    }

    vector_t& vector_t::operator=(const vector_t& that) {
        if (this == &that) {}
        assign(that);
        return self;
    }

    vector_t& vector_t::operator=(real_t s) {
        fill(s);
        return self;
    }

}