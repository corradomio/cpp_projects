//
// Created by Corrado Mio on 26/03/2022.
//

#include <igraph.h>
#include "igraph.hpp"
#define self (*this)

namespace ig = igraph;
namespace igs = igraph::simple;

// --------------------------------------------------------------------------

void ig::vector_t::release() const {
    if (ig::refcount_t::release()) {
        igraph_vector_destroy(self.v);
        igraph_free(self.v);
        self.v = nullptr;
    }
}

// --------------------------------------------------------------------------

ig::vector_t::vector_t(int n) {
    self.v = (igraph_vector_p) igraph_malloc(sizeof(igraph_vector_t));
    if (n >= 0)
        check(igraph_vector_init(self.v, n));
}
