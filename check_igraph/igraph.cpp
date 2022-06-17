//
// Created by Corrado Mio on 25/03/2022.
//

#include <igraph.h>
#include "igraph.hpp"
#define self (*this)

// --------------------------------------------------------------------------

namespace ig = igraph;

const char* ig::exception::what() const noexcept {
    return "";
}

void ig::check(int ecode) {
    if (ecode != IGRAPH_SUCCESS)
        throw ig::exception(ecode);
}

// --------------------------------------------------------------------------
// reference counting
// --------------------------------------------------------------------------

ig::refcount_t::refcount_t(){
    self.c = (count_p )igraph_malloc(sizeof(count_t));
    self.c->count = 1;
}

bool ig::refcount_t::release() const {
    bool none = 0 == (self.c->count -= 1);
    if (none) {
        igraph_free(self.c);
        self.c = nullptr;
    }
    return none;
}

void ig::refcount_t::add_ref() const {
    self.c->count += 1;
}

void ig::graph_t::release() const {
    if (ig::refcount_t::release()) {
        igraph_destroy(self.g);
        igraph_free(self.g);
        self.g = nullptr;
    }
}

// --------------------------------------------------------------------------
// Constructor
// --------------------------------------------------------------------------

ig::graph_t::graph_t(int n, bool directed) {
    self.g = (igraph_p)igraph_malloc(sizeof(::igraph_t));
    if (n >= 0)
        check(::igraph_empty(self.g, n, directed ? IGRAPH_DIRECTED : IGRAPH_UNDIRECTED));
}

ig::graph_t::graph_t(const graph_t& g) {
    g.add_ref();
    self.c = g.c;
    self.g = g.g;
}

ig::graph_t::~graph_t() {
    release();
}

ig::graph_t& ig::graph_t::operator =(const graph_t& g) {
    g.add_ref();
    self.release();
    self.c = g.c;
    self.g = g.g;
    return self;
}

ig::graph_t ig::graph_t::clone() const {
    graph_t copy;
    check(igraph_copy(copy.g, self.g));
    return copy;
}

// --------------------------------------------------------------------------

int ig::graph_t::order() const {
    return igraph_vcount(self.g);
}

int ig::graph_t::size() const {
    return igraph_ecount(self.g);
}

bool ig::graph_t::directed() const {
    return igraph_is_directed(self.g);
}


// --------------------------------------------------------------------------

ig::edge_t ig::graph_t::get_edge(int eid) {
    ig::edge_t e;
    check(::igraph_edge(self.g, eid, &e.source, &e.target));
    return e;
}

