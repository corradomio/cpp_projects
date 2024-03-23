//
// Created by Corrado Mio on 08/03/2024.
//

#include "stdx/float64/array.h"

namespace stdx::float64 {

    // ----------------------------------------------------------------------

    void array_t::alloc(size_t n) {
        self._info = (info_t*)new char[sizeof(info_t) + n * sizeof(real_t)];
        self._info->refc = 1;
        self._info->n = n;
        self._data = self._info->data;
    }

    void array_t::fill(real_t s) {
        size_t n=self.size();
        for(size_t i=0; i<n; ++i) self._data[i] = s;
    }

    void array_t::init(const array_t& that) {
        self._info = that._info;
        self._data = that._data;
        self.add_ref();
    }

    void array_t::assign(const array_t& that) {
        that.add_ref();
        self.release();
        self._info = that._info;
        self._data = that._data;
    }

    void array_t::fill(const array_t& that) {
        size_t n = self.size();
        for(size_t i=0; i<n; ++i) self._data[i] = that._data[i];
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
