//
// Created by Corrado Mio on 07/03/2024.
//

#include "array.h"

#ifndef STDX_FLOAT64_VECTOR_H
#define STDX_FLOAT64_VECTOR_H


namespace stdx::float64 {

    struct vector_t;

    // struct vector_it {
    //     using iterator_category = std::forward_iterator_tag;
    //     using difference_type   = std::ptrdiff_t;
    //     using value_type        = Float;
    //     using pointer           = Float*;  // or also value_type*
    //     using reference         = Float&;  // or also value_type&
    //
    //     const vector_t& v;
    //     size_t at;
    //
    //     vector_it(const vector_t& v, size_t at): v(v), at(at) { };
    //     bool operator==(const vector_it& it) const { return self.at == it.at; }
    //     bool operator!=(const vector_it& it) const { return self.at != it.at; }
    //     vector_it& operator++() { ++at; return self; }
    //     reference operator *() const; // { return v[at]; }
    // };

    struct vector_t : public array_t {
        using super = array_t;

        // create by ref
        void init(const vector_t& that);
        // assign by ref
        void assign(const vector_t& that);
        // copy the content
        void fill(real_t s);
        void fill(const vector_t& that);

        // ------------------------------------------------------------------
        // Constructors

        explicit vector_t(size_t n=0): super(n) { };
        vector_t(const vector_t& that, bool clone=false): super(that, clone) { }

        // ------------------------------------------------------------------
        // References

        [[nodiscard]] vector_t  clone() const { return {self, true}; }
        [[nodiscard]] vector_t norefs() const { return self.info->refc==1 ? self : self.clone(); }

        // ------------------------------------------------------------------
        // Assignment

        vector_t& operator=(const vector_t& v);
        vector_t& operator=(real_t s);

        // ------------------------------------------------------------------
        // Accessors

        // Float& operator[](size_t i)       { return self.data[i]; }
        real_t& operator[](size_t i) const {
            // assert(i < self.size());
            return self.data[i];
        }

        // ------------------------------------------------------------------
        // Iterator

        array_it begin() { return array_it{self, 0, 1}; }
        array_it   end() { return array_it{self, self.size(), 0}; }

        // End
        // ------------------------------------------------------------------

    };

}

#endif //STDX_FLOAT64_VECTOR_H
