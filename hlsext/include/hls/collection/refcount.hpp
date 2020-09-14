//
// Created by Corrado Mio on 11/05/2015.
//

#ifndef REFCOUNT_HPP
#define REFCOUNT_HPP

#include <stddef.h>
#include <typeinfo>
#include <assert.h>

namespace hls {
namespace collection {

    class refcount_t {
    public:
        mutable size_t count;
    public:
        refcount_t() : count(0) { }
        virtual ~refcount_t(){ assert(count == 0); }
    };


    template<typename T>
    class ref_ptr {
        T* ptr;

        void add_ref() const { if (ptr) ++ptr->count; }
        void release() { if (ptr && 0 == --ptr->count) { delete ptr; ptr = 0; } }
    public:
        ref_ptr() : ptr(0) { }
        ref_ptr(T* p) : ptr(p) { add_ref(); }
        ref_ptr(const ref_ptr& rp) : ptr(rp.ptr) { add_ref(); }
       ~ref_ptr() { release(); }

        ref_ptr& operator =(const ref_ptr& rp) {
            rp.add_ref();
            release();
            ptr = rp.ptr;
            return *this;
        }

        ref_ptr& operator =(T* p) {
            if(p) ++p->count;
            release();
            ptr = p;
            return *this;
        }

        operator T*() const { return ptr; }

        T* operator->() const { return  ptr; }
        T& operator *() const { return *ptr; }

        T* get() const { return ptr; }
        T* detach() { T* p = ptr; ptr = 0; return p; }

    };

    template<typename T, typename R>
    T* ref_cast(const ref_ptr<R>& rp) {
        R* r = rp.get();
        T* p = dynamic_cast<T*>(r);
        if (r && !p) throw std::bad_cast();
        return p;
    }

}}


#endif //REFCOUNT_HPP
