//
// Created by Corrado Mio on 08/03/2024.
//

#ifndef STDX_FLOAT64_ARRAY_H
#define STDX_FLOAT64_ARRAY_H

#include <cassert>
#include <language.h>
#include <stdx/exceptions.h>
#include "arith.h"


namespace stdx::float64 {

    struct array_t {

        struct info_t {
            size_t refc;
            size_t n;
            real_t data[0];
        };

        info_t* _info;
        real_t* _data;   // same as info->data

        void alloc(size_t n);
        void add_ref() const { self._info->refc++; }
        void release() const { if (0 == --self._info->refc) delete (char*)self._info; }

        // create by ref
        void init(const array_t& that);
        // assign by ref
        void assign(const array_t& that);

        void fill(real_t s);
        void fill(const array_t& that);

        // ------------------------------------------------------------------
        // Constructors

        explicit array_t(size_t n);
        array_t(const array_t& that, bool clone=false);
        ~array_t() { release(); }

        // ------------------------------------------------------------------
        // Properties

        /// n of elements in the array
        [[nodiscard]] size_t size() const { return self._info->n; }
        /// pointer to the first element of the array
        [[nodiscard]] real_t* data() const { return self._data; }

        // ------------------------------------------------------------------
        // Accessors

        real_t& operator[](size_t i) const { return self._data[i]; }

    };

    /**
     * Array iterator
     */
    struct array_it {
        using value_type        = real_t;
        using iterator_category = std::forward_iterator_tag;
        using difference_type   = std::ptrdiff_t;
        using pointer           = value_type*;
        using reference         = value_type&;

        array_t& a;
        size_t at;
        size_t step;

        array_it(array_t& a, size_t at=0, size_t step=1) : a(a), at(at), step(step) { }
        bool operator==(const array_it& it) const { return self.at == it.at; }
        bool operator <(const array_it& it) const { return self.at  < it.at; }
        array_it& operator++() { at += step; return self; }
        reference operator *() const { return a[at]; }

    };

    /**
     * Style used to dump a vector/matrix"
     *
     *       PYTHON: [...]
     *  MATHEMATICA: {...}
     */
    enum array_style {
        PYTHON,
        MATHEMATICA
    };
};


#endif //STDX_FLOAT64_ARRAY_H
