//
// Created by Corrado Mio on 16/12/2023.
//

#ifndef REFC_H
#define REFC_H

#include <cstddef>

namespace stdx {

    struct refcount_t {
        size_t refc;

        refcount_t(): refc(0) {}
        virtual ~refcount_t() { }

        inline static void add_ref(refcount_t *p) {
            if (p != nullptr)
                p->refc++;
        }
        inline static void release(refcount_t *p) {
            if (p != nullptr && 0 == --(p->refc)) {
                delete p;
            }
        }
    };


    template<typename T>
    struct refp {
        T* _ptr;

        refp(): _ptr(nullptr){}
        refp(T* p): _ptr(p){ refcount_t::add_ref(_ptr); }
        refp(const refp<T>& rp): _ptr(rp._ptr){ refcount_t::add_ref(_ptr); }
        ~refp(){ refcount_t::release(_ptr); _ptr = nullptr; }

        refp& assign(T* p) {
            refcount_t::add_ref(p);
            refcount_t::release(_ptr);
            _ptr = p;
            return *this;
        }

        refp& operator =(T* p) {
            return assign(p);
        }

        refp& operator =(const refp<T>& rp) {
            return assign(rp.get());
        }

        T* get()               const { return _ptr; }
        explicit operator T*() const { return _ptr; }
    };

    // static_cast
    // dynamic_cast
    // reinterpret_cast
    // const_cast

    template<typename R, typename T>
    inline R* ref_cast(const refp<T>& rp) {
        T* t = rp.get();
        R* r = dynamic_cast<R*>(t);
        if (r == nullptr && t != nullptr)
            throw std::bad_cast();
        return r;
    }
}

#endif //REFC_H
