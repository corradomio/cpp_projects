//
// Created by Corrado Mio on 06/01/2024.
//

#ifndef CHECK_GSL_GSL_CPP_H
#define CHECK_GSL_GSL_CPP_H

#include <stdlib.h>
#include <assert.h>

struct refcount {
    size_t ref;
    refcount(): ref(0) {}
    virtual ~refcount(){ assert(this->ref == 0); }
};

struct refp {

    inline static void add_ref(refcount *p) {
        if (p != nullptr)
            p->ref++;
    }

    inline static void release(refcount *p) {
        if (p != nullptr && 0 == --(p->ref))
            delete p;
    }

    inline void assign(refcount *p) {
        add_ref(p);
        release(ptr);
        ptr = p;
    }

    refcount *ptr;
    refp(): ptr(nullptr) {}
    explicit refp(refcount *p)  : ptr(p) { add_ref(ptr); }
    ~refp(){ release(ptr); ptr = nullptr; }

};

namespace gsl {
    template<typename T>

}

#endif //CHECK_GSL_GSL_CPP_H
