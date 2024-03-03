//
// Created by Corrado Mio on 11/02/2024.
//
#include "octree.h"
#include <stdio.h>

using namespace stdx;
using namespace stdx::octree;


// --------------------------------------------------------------------------
// octree_t
// --------------------------------------------------------------------------

node_t* octree_t::select(const stdx::v3d_t &p, size_t n)  {
    node_t* c = r;
    node_t* s = nullptr;
    while (c != nullptr) {
        s = c;
        c = c->select(p);
    }
    if (n > 0 && s->r.n > n) {
        s = s->split(p, m);
    }
    return s;
}

size_t octree_t::add(const element_t& e) {
    size_t id = m.add(e);
    node_t* node = select(e.p, n);
    node->r.add(id);
    return id;
}


void octree_t::save(const char* fname) {
    FILE *f = fopen(fname,"w");
    fprintf(f, "x,y,z,m\n");
    for (int i=0; i<size(); ++i) {
        element_t& e = m[i];
        fprintf(f, "%.6f,%.6f,%.6f,%.6f\n",e.p.x, e.p.y, e.p.x, e.m);
    }
    fclose(f);
}

// --------------------------------------------------------------------------
// node_t
// --------------------------------------------------------------------------

bool node_t::contains(const v3d_t& p) const {
    return l.x < p.x && p.x <= u.x &&
           l.y < p.y && p.y <= u.y &&
           l.z < p.z && p.z <= u.z;
}


node_t* node_t::select(const v3d_t& p) {
    size_t n = c.n;
    for (int i=0; i<n; ++i)
        if (c[i].contains(p))
            return &(c[i]);
    return nullptr;
}


node_t* node_t::split(const v3d_t& p, array_t<element_t>& m) {
    v3d_t t = l + (u-l)/2;

    // split the current cube in 8 sub-cubes
    c.allocate(8);
    for(int i=0; i<8; ++i) {
        c[i].p = this;

        // x
        if ((i&1) == 0) {
            c[i].l.x = l.x;
            c[i].u.x = t.x;
        } else {
            c[i].l.x = t.x;
            c[i].u.x = u.x;
        }
        // y
        if ((i&2) == 0) {
            c[i].l.y = l.y;
            c[i].u.y = t.y;
        } else {
            c[i].l.y = t.y;
            c[i].u.y = u.y;
        }
        // z
        if ((i&4) == 0) {
            c[i].l.z = l.z;
            c[i].u.z = t.z;
        } else {
            c[i].l.z = t.z;
            c[i].u.z = u.z;
        }
    }

    // distribute the current members into the sub cubes
    for (int i=0; i<r.n; ++i) {
        size_t id = r[i];
        const element_t& e = m[id];
        node_t* s = select(e.p);
        s->add(id);
    }

    clear();
    node_t* s = select(p);
    return s;
}


void spaces(int n) {
    while(n-- > 0) printf("   ");
}


void node_t::dump(int depth) {
    spaces(depth); printf("[%3d: (%.3f,%.3f,%.3f), (%.3f,%.3f,%.3f)]\n", depth, l.x, l.y, l.z, u.x, u.y, u.z);
    spaces(depth); printf("   n: %d\n", r.n);
    for(int i=0; i<c.n; ++i)
        c[i].dump(depth+1);
}
