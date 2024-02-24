//
// Created by Corrado Mio on 15/02/2024.
//

#ifndef CHECK_NBODY_SPACE_H
#define CHECK_NBODY_SPACE_H

#include "array.h"
#include "v3d.h"


namespace space {

    struct object_t {
        float m;    // mass
        v3d p;      // position
        v3d v;      // velocity
        v3d a;      // acceleration

        object_t() {}

        object_t(float m) : m(m) {}

        object_t(float m, const v3d &p, const v3d &v) : m(m), p(p), v(v) {}

        // object_t(const object_t& o): m(o.m), p(o.p), v(o.v) { }
        object_t(const object_t &o) = default;

        /*object_t& operator =(const object_t& o) {
            m = o.m;
            p = o.p;
            v = o.v;
            return self;
        }*/
        object_t &operator=(const object_t &o) = default;
    };


    struct space_t {
        /// center of mass
        object_t cmass;
        /// members
        stdx::array_t <object_t> members;

        space_t(int s) : members(s) {}

        /// Compute the reference point (centro di massa)
        ///
        void make_cmass(bool stationary);
    };

}

#endif //CHECK_NBODY_SPACE_H
