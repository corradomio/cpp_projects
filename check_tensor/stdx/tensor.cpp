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

    shape_t::shape_t(): rank(0) {
            dims[0] = 0;
    }

    // shape_t::shape_t(uint16 r): rank(r) {
    //     for (uint16 i=0; i<=r; ++i)
    //         dims[i] = 0;
    // }

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

    rank_t::rank_t(const shape_t& shape)
    // : rank_t(shape.rank)
    : r(shape.rank)
    {
        for(uint16 i = 0; i < shape.rank; ++i) {
            uint16 sz = shape[i];
            self.dims[i] = {sz };
            for (int d=i-1; d >=0; --d)
                self.dims[d].esize *= sz;
            self.dord[i] = i;
        }
        self.dims[self.r] = {1};
    }

    // ---

    // size_t rank_t::size_() const {
    //     size_t sz = 1;
    //     for (uint16 i=0; i < self.r; ++i)
    //         sz *= self.dims[i].dlen;
    //     return sz;
    // }

    // ---

    uint16 rank_t::rank() const {
        // uint16 r = 0;
        // for (uint16 i=0; i< self.r; ++i)
        //     if (self.dim(i) > 1)
        //         r++;
        // return r;
        return r;
    }

    uint16 rank_t::dim(uint16 i) const {
        uint16 d = self.dord[i];
        return self.dims[d].len/self.dims[d].step;
    }

    uint16 rank_t::dim_(uint16 i) const {
        uint16 d = self.dord[i];
        return self.dims[d].dlen;
    }

    size_t rank_t::size() const {
        size_t sz = 1;
        for (uint16 i=0; i<self.r; ++i)
            sz *= self.dim(i);
        return sz;
    }

    shape_t rank_t::shape() const {
        shape_t shape;
        for (uint16 i=0; i<self.r; ++i) {
            uint16 d = self.dord[i];
            shape.dims[i] = self.dim(d);
        }
        shape.rank = self.r;
        return shape;
    }

    // ---

    void rank_t::swap(uint16 d1, uint16 d2) {
        std::swap(self.dord[d1], self.dord[d2]);
    }

    void rank_t::swap(const std::initializer_list<uint16>& dorder) {
        uint16 i = 0;
        int consistency = 0;

        assert (dorder.size() == self.r);

        for (auto it= dorder.begin(); it != dorder.end(); ++it, ++i) {
            uint16 d = *it;
            self.dord[i] = d;
            consistency += (i-d);
        }

        assert (consistency == 0);
    }

    // ---

    size_t rank_t::at(const std::initializer_list<uint16>& indices) const {
        assert(indices.size() == self.r);
        size_t at = 0;
        uint16 i  = 0;
        uint16 index;
        for (auto it= indices.begin(); it!=indices.end(); ++it,++i) {
            uint16 d = self.dord[i];
            const dim_t& dim = self.dims[d];
            index = *it;
            at += dim.doff + (dim.off + index*dim.step)*dim.esize;
        }
        {
            const dim_t& dim = self.dims[i];
            at += dim.doff + (dim.off + 0*dim.step)*dim.esize;
        }
        return at;
    }

    void rank_t::view(const std::initializer_list<span_t>& spans) {
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
        for(auto it= spans.begin(), itend= spans.end();
            it != itend;
            ++it, ++i)
        {
            dim_t& dim  = self.dims[i];
            span_t span = *it;
            uint16 dim_step = dim.step;

            if (span.off == ANY && span.end == 0)
                span.off = 0, span.end = dim.len/dim_step;
            if (span.off == ANY)
                span.off = 0;
            if (span.end == ANY)
                span.end = dim.len/dim_step;

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


    void rank_t::compact() {
        uint16 i = 0;
        uint16 rank = self.r;
        while (i < rank) {
            const dim_t& dim = self.dims[i];
            if (dim.len == 1) {
                size_t doff = dim.doff + dim.off*dim.esize;
                self.dims[i+1].doff += doff;
                ::memcpy(&self.dims[i], &self.dims[i+1], (rank-i)*sizeof(dim_t));
                rank -= 1;
                self.r = rank;
            }
            else {
                i++;
            }
        }
    }

    // ----------------------------------------------------------------------
    // end
    // ----------------------------------------------------------------------

}
