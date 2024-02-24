//
// Created by Corrado Mio on 11/02/2024.
//

#ifndef CHECK_STRUCTURES_OCTREE_H
#define CHECK_STRUCTURES_OCTREE_H

#include "array.h"
#include "v3d.h"

namespace stdx::octree {

    struct element_t {
        float m;
        v3d_t p;
        v3d_t v;
        v3d_t a;
    };

    struct node_t {
        v3d_t l,u;          // lower, upper
        array_t<size_t> r;  // references

        node_t* p;          // parent
        array_t<node_t> c;  // children

        node_t(): l(),u(), r(), c(), p(nullptr) { }
        node_t(const v3d_t& l, const v3d_t& u): l(l), u(u), r(), c() { }
        ~node_t() { }

        void add(size_t id) { r.add(id); }
        void clear() { r.clear(); }

        float length() const { return u.x - l.x; }

        [[nodiscard]] bool  contains(const v3d_t& p) const;
        [[nodiscard]] node_t* select(const v3d_t& p);
        [[nodiscard]] node_t*  split(const v3d_t& p, array_t<element_t>& m);

        void dump(int depth);
    };

    struct octree_t {
        array_t<element_t>  m;   // members
        node_t*             r;   // root
        size_t              n;   // max size

        octree_t(float l, size_t n_) {
            n = n_;
            r = new node_t(v3d_t(0,0,0),v3d_t(l,l,l));
        }
        octree_t(const v3d_t& l, const v3d_t& u, size_t n_) {
            n = n_;
            r = new node_t(l,u);
        }
        ~octree_t() { delete r; }

        [[nodiscard]] size_t size() const { return m.size(); }

        node_t* select(const v3d_t& p, size_t n=0);
        size_t  add(const element_t& e);

        void dump() { r->dump(0); }

        void save(const char* fname);
    };

}

#endif //CHECK_STRUCTURES_OCTREE_H
