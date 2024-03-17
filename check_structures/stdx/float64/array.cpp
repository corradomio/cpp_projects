//
// Created by Corrado Mio on 08/03/2024.
//

#include "array.h"

namespace stdx::float64 {

    // ----------------------------------------------------------------------

    void array_t::alloc(size_t n) {
        self.info = (info_t*)new char[sizeof(info_t) + n*sizeof(real_t)];
        self.info->refc = 1;
        self.info->n = n;
        self.data = self.info->data;
    }

    void array_t::fill(real_t s) {
        size_t n=self.size();
        for(size_t i=0; i<n; ++i) self.data[i] = s;
    }

    void array_t::init(const array_t& that) {
        self.info = that.info;
        self.data = that.data;
        self.add_ref();
    }

    void array_t::assign(const array_t& that) {
        that.add_ref();
        self.release();
        self.info = that.info;
        self.data = that.data;
    }

    void array_t::fill(const array_t& that) {
        size_t n = self.size();
        for(size_t i=0; i<n; ++i) self.data[i] = that.data[i];
    }

    // ----------------------------------------------------------------------

    array_t::array_t(size_t n) {
        alloc(n);
    }

    array_t::array_t(const array_t& that, bool clone) {
        if (clone) {
            alloc(that.size());
            fill(that);
        }
        else {
            init(that);
        }
    }

}
