//
// Created by Corrado Mio on 25/03/2022.
//
#include <exception>
#include <vector>
#include <igraph.h>

#ifndef CHECK_IGRAPH_IGRAPH_HPP
#define CHECK_IGRAPH_IGRAPH_HPP

typedef ::igraph_t *igraph_p;
typedef ::igraph_vector_t *igraph_vector_p;
typedef struct count_t { int count; }* count_p;

namespace igraph {

    class exception : public std::exception {
        int ecode;
    public:
        exception(int ecode): ecode(ecode) {}

        virtual const char* what() const noexcept;
    };

    void check(int ecode);

    class refcount_t {
    protected:
        mutable count_p c;
        bool release() const;
        void add_ref() const;

        refcount_t();
    };

}


namespace igraph {

    struct graph_t;
    struct edge_t;

    struct vector_t : refcount_t {
        mutable igraph_vector_p v;

        void release() const;
    public:
        vector_t(): vector_t(-1) {}
        vector_t(int n);
        vector_t(const vector_t& v);
        ~vector_t();

        vector_t& operator =(const vector_t& v);
    };


    struct graph_t : refcount_t {
        mutable igraph_p g;

        void release() const;
    public:
        // -------------------------------------------------------------------
        // Constructors
        // -------------------------------------------------------------------

        /// uninitialized graph
        graph_t(): graph_t(-1, false) {}
        /// empty undirected graph with n vertices
        graph_t(int n): graph_t(n, false) {}
        /// empty (un)directed graph with n vertices
        graph_t(int n, bool directed);
        /// copy the graph
        graph_t(const graph_t& g);
        /// destroy the graph
       ~graph_t();

        // -------------------------------------------------------------------
        // Assignment
        // -------------------------------------------------------------------

        /// delete the current graph and copy the graph passed
        /// as argument
        graph_t& operator =(const graph_t& g);

        graph_t clone() const;

        // -------------------------------------------------------------------
        // Properties
        // -------------------------------------------------------------------

        /// directed
        bool directed() const;

        /// n of vertices
        int order() const;
        /// n of edges
        int size() const;

        // -------------------------------------------------------------------
        // Edges
        // -------------------------------------------------------------------

        edge_t get_edge(int eid);

        // -------------------------------------------------------------------
        // End
        // -------------------------------------------------------------------

    };

    struct edge_t {
        igraph_integer_t source;
        igraph_integer_t target;
    };

}

namespace igraph { namespace simple {

    graph_t from_edges(const std::vector<edge_t>& edges, bool directed);
}}

#endif //CHECK_IGRAPH_IGRAPH_HPP
