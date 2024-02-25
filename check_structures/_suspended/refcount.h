//
// Created by Corrado Mio on 10/01/2024.
//

#ifndef REFCOUNT_H
#define REFCOUNT_H

#include <cstdlib>
#include <typeinfo>

namespace stdx {

    struct refc_t {
        size_t refc;

        refc_t(): refc(0) { }
        virtual ~refc_t() { }
    };

    template<typename T>
    struct ref_ptr {
        T* ptr;

        inline void add_ref(refc_t* p) {
            if (p != nullptr)
                p->refc++;
        }

        inline void release(refc_t* p) {
            if (p != nullptr && 0 == (--p->refc))
                delete p;
        }

        inline void assign(refc_t* p) {
            add_ref(p);
            release(ptr);
            ptr = dynamic_cast<T*>(p);
            if (p != nullptr && ptr == nullptr)
                throw std::bad_cast();
        }

        ref_ptr(): ptr(nullptr) {}
        ref_ptr(T* p): ptr(p) { add_ref(ptr); }
        ref_ptr(const ref_ptr& rp): ptr(rp.ptr) { add_ref(ptr); }

        virtual ~ref_ptr() { release(ptr); }

        ref_ptr& operator =(T* p) {
            assign(p);
            return *this;
        }

        ref_ptr& operator =(const ref_ptr& rp) {
            assign(rp.ptr);
            return *this;
        }

        // T* get() const { return  ptr; }
        // T& ref() const { return *ptr; }

        T* operator->() const { return  ptr; }
        T& operator *() const { return *ptr; }
    };

    template<typename R, typename T>
    ref_ptr<R> ref_cast(const ref_ptr<T>& rp) {
        T* pt = rp.ptr;
        R* pr = dynamic_cast<R*>(pt);
        if (pt != nullptr && pr == nullptr)
            throw std::bad_cast();
        return ref_ptr<R>(pr);
    }

}

#endif //REFCOUNT_H
