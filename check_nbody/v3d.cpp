//
// Created by Corrado Mio on 07/02/2024.
//
#include <cmath>
#include "v3d.h"


inline float sq(float x){ return x*x;   }
inline float cubic(float x){ return x*x*x; }


/// compute the total mass, the center of mass and the
/// global velocity.
/// :param stationary: if the object must have 0 global velocity
void cluster_t::make_cmass(bool stationary) {
    float mass;
    object_t& ref = self.cmass;
    object_t* elts = self.members.data;
    ref.p = 0;
    ref.m = 0;

    int n = self.members.n;
    for(int i=0; i<n; ++i) {
        mass = elts[i].m;
        ref.p.linear(mass, elts[i].p);
        ref.v.linear(mass, elts[i].v);
        ref.m += mass;
    }
    ref.p /= ref.m;
    ref.v /= ref.m;

    if (stationary) {
        for(int i=0; i<n; ++i) {
            elts[i].v -= ref.v;
        }
    }
}