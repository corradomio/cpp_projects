//
// Created by Corrado Mio on 11/05/2015.
//

#ifndef REFCOUNT_HPP
#define REFCOUNT_HPP

#include <stddef.h>
#include <typeinfo>
#include <assert.h>

namespace mem {

    template<typename _Tp>
    class ref_ptr {
        mutable size_t *pcnt;
        _Tp *ptr;
    public:
        ref_ptr() : ptr(nullptr), pcnt(new size_t(1)) {
            *pcnt = 0;
        }
        ref_ptr(_Tp* ptr) : ptr(ptr), pcnt(new size_t) {
            *pcnt = (ptr == nullptr) ? 0 : 1;
        }
        ref_ptr(const ref_ptr& p) : ptr(p.ptr), pcnt(p.pnct) {
            if (ptr != nullptr)
                *pcnt += 1;
        }

        ~ref_ptr() {
            if (--*pcnt > 0)
                return;
            delete pcnt;
            delete ptr;
        }
    };



    //class refcount_t {
    //public:
    //    mutable size_t count;
    //public:
    //    refcount_t() : count(0) { }
    //    virtual ~refcount_t(){ assert(count == 0); }
    //};

    //template<typename T>
    //class ref_ptr {
    //    T* ptr;
    //
    //    void add_ref() const { if (ptr) ++ptr->count; }
    //    void release() { if (ptr && 0 == --ptr->count) { delete ptr; ptr = nullptr; } }
    //public:
    //    ref_ptr() : ptr(nullptr) { }
    //    ref_ptr(T* p) : ptr(p) { add_ref(); }
    //    ref_ptr(const ref_ptr& rp) : ptr(rp.ptr) { add_ref(); }
    //   ~ref_ptr() { release(); }
    //
    //    ref_ptr& operator =(const ref_ptr& rp) {
    //        rp.add_ref();
    //        release();
    //        ptr = rp.ptr;
    //        return *this;
    //    }
    //
    //    ref_ptr& operator =(T* p) {
    //        if(p) ++p->count;
    //        release();
    //        ptr = p;
    //        return *this;
    //    }
    //
    //    operator T*() const { return ptr; }
    //
    //    T* operator->() const { return  ptr; }
    //    T& operator *() const { return *ptr; }
    //
    //    T* get() const { return ptr; }
    //    T* detach() { T* p = ptr; ptr = nullptr; return p; }
    //
    //};

    //template<typename T, typename R>
    //T* ref_cast(const ref_ptr<R>& rp) {
    //    R* p = rp.get();
    //    T* t = dynamic_cast<T*>(p);
    //    if (t && !p) throw std::bad_cast();
    //    return t;
    //}

}


#endif //REFCOUNT_HPP
