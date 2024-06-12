//
// Created by Corrado Mio on 10/06/2024.
//

#include <initializer_list>
#include "language.h"
#include <cassert>
#include "tensor.h"

namespace stdx::linalg {

    // ----------------------------------------------------------------------
    // shape_t
    // ----------------------------------------------------------------------

    shape_t::shape_t(uint16 r): rank(r) {
        for (uint16 i=0; i<=r; ++i)
            dims[i] = 0;
    }

    shape_t::shape_t(const std::initializer_list<uint16>& dims)
    : rank((uint16)dims.size())
    {
        uint16 i=0;
        for (auto it= dims.begin(); it != dims.end(); ++it, ++i)
            self.dims[i] = *it;
        self.dims[i] = 0;
    }

    size_t shape_t::size() const {
        size_t sz = 1;
        for (size_t  i=0; i<self.rank; ++i)
            sz *= self.dims[i];
        return sz;
    }

    // ----------------------------------------------------------------------
    // rank_t
    // ----------------------------------------------------------------------

    rank_t::rank_t(uint16 r): r(r)/*, dims(new dim_t[r+1])*/ {
        for(uint16 i=0; i<=r; ++i)
            self.dims[i] = {0};
    }

    rank_t::rank_t(const shape_t& shape): rank_t(shape.rank) {
        for(uint16 i = 0; i < shape.rank; ++i) {
            uint16 sz = shape[i];
            self.dims[i] = {sz };
            for (int d=i-1; d >=0; --d)
                self.dims[d].dlen *= sz;
        }
    }

    // ---

    size_t rank_t::size_() const {
        size_t sz = 1;
        for (uint16 i=0; i< self.r; ++i)
            sz *= self.dims[i].dlen;
        return sz;
    }

    // ---

    uint16 rank_t::rank() const {
        uint16 r = 0;
        for (uint16 i=0; i< self.r; ++i)
            if (self.dim(i) > 1)
                r++;
        return r;
    }

    uint16 rank_t::dim(uint16 i) const {
        return self.dims[i].len/self.dims[i].step;
    }

    size_t rank_t::size() const {
        size_t sz = 1;
        for (uint16 i=0; i<self.r; ++i)
            sz *= self.dim(i);
        return sz;
    }

    shape_t rank_t::shape() const {
        shape_t shape(self.rank());
        for (uint16 i=0, j=0; i<self.r; ++i)
            if (self.dim(i) > 0)
                shape.dims[j++] = self.dim(i);
        return shape;
    }

    // ---

    size_t rank_t::at(const std::initializer_list<uint16>& indices) const {
        assert(indices.size() == self.r);
        size_t sz  = self.size_();
        size_t at  = 0;
        uint16 i   = 0;
        for (auto it=indices.begin(); it!=indices.end(); ++it,++i) {
            const dim_t& dim = self.dims[i];
            uint16 index = *it;
            sz /=  dim.dlen;
            at += (dim.off + index*dim.step)*sz;
        }
        return at;
    }

    void rank_t::sel(const std::initializer_list<span_t>& spans) {
        assert(spans.size() <= self.r);

        //   i
        //  -i
        //
        //   i;;j
        //   i;;ALL
        // ALL;;j
        // ALL;;ALL
        //
        //   i;;  j;;k
        //   i;;ALL;;k
        // ALL;;  j;;k
        // ALL;;ALL;;k
        // ------------------------------------

        uint16 i = 0;
        for(auto it= spans.begin(), itend=spans.end(); it != itend; ++it, ++i) {
            dim_t& dim  = self.dims[i];
            span_t span = *it;
            uint16 dim_step = dim.step;

            if (span.off == ANY && span.end == 0)
                span.off = 0, span.end = dim.len/dim_step;
            if (span.off == ANY)
                span.off = 0;
            if (span.end == ANY)
                span.end = dim.len/dim_step;

            assert(span.step > 0);

            dim.off  +=  span.off*dim_step;
            dim.len   = (span.end-span.off)*dim_step;
            dim.step *=  span.step;

            // consistency checks:
            if (dim.off > dim.dlen)
                dim.off = dim.dlen;
            if ((dim.off+dim.len) > dim.dlen)
                dim.len = dim.dlen - dim.off;
        }
    }

    // ----------------------------------------------------------------------
    // end
    // ----------------------------------------------------------------------

}
