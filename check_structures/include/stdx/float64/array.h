//
// Created by Corrado Mio on 08/03/2024.
//

#ifndef STDX_FLOAT64_ARRAY_H
#define STDX_FLOAT64_ARRAY_H

#include <cassert>
#include "../language.h"
#include "../exceptions.h"


namespace stdx::float64 {

    typedef double real_t;

    struct array_t {

        struct info_t {
            size_t refc;
            size_t n;
            real_t data[0];
        };

        info_t* info;
        real_t* data;

        void alloc(size_t n);
        void add_ref() const { self.info->refc++; }
        void release() const { if (0 == --self.info->refc) delete (char*)self.info; }

        // create by ref
        void init(const array_t& that);
        // assign by ref
        void assign(const array_t& that);

        void fill(real_t s);
        void fill(const array_t& that);

        // ------------------------------------------------------------------
        // Constructors

        array_t(size_t n);
        array_t(const array_t& that, bool clone=false);

        ~array_t() { release(); }

        // ------------------------------------------------------------------
        // Properties

        [[nodiscard]] size_t size() const { return self.info->n; }

        // ------------------------------------------------------------------
        // Accessors

        // Float& operator[](size_t i)       { return self.data[i]; }
        real_t& operator[](size_t i) const {
            // assert(i < self.size());
            return self.data[i];
        }

    };

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

};


#endif //STDX_FLOAT64_ARRAY_H
