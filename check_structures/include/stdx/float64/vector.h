//
// Created by Corrado Mio on 07/03/2024.
//

#include "array.h"

#ifndef STDX_FLOAT64_VECTOR_H
#define STDX_FLOAT64_VECTOR_H


namespace stdx::float64 {

    struct vector_t;

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

        real_t& operator[](size_t i) const { return self.data[i]; }

        // ------------------------------------------------------------------
        // Iterator

        array_it begin() { return array_it{self, 0, 1}; }
        array_it   end() { return array_it{self, self.size(), 0}; }

        // End
        // ------------------------------------------------------------------

    };

}

#endif //STDX_FLOAT64_VECTOR_H
