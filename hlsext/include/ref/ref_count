//
// Created by Corrado Mio on 11/05/2015.
//

#ifndef REFCOUNT_HPP
#define REFCOUNT_HPP

#include <stddef.h>
#include <typeinfo>
#include <assert.h>

namespace ref {

    template<typename T>
    class ref_ptr {
        mutable size_t *_cnt;
        T *_ptr;

        void add_ref(ref_ptr& rp) {
            if (_ptr != nullptr) (*_cnt)++;
        }

        void release(ref_ptr& rp) {
            if (_ptr != nullptr && --(*_cnt) == 0) {
                delete _cnt;
                delete _ptr;
            }
        }
    public:
        typedef T      * pointer;
        typedef T const* const_pointer;
        typedef T      & reference;
        typedef T const& const_reference;
    public:
        explicit ref_ptr() : _ptr(nullptr), _cnt(new size_t(0)) {
            add_ref(*this);
        }

        explicit ref_ptr(pointer ptr) : _ptr(ptr), _cnt(new size_t(0)) {
            add_ref(*this);
        }

        ref_ptr(const_reference p) : _ptr(p._ptr), _cnt(p._cnt) {
            add_ref(*this);
        }

        ref_ptr& operator =(const_reference p) {
            add_ref(p);
            release(*this);
            _cnt = p._cnt;
            _ptr = p._ptr;
            return *this;
        }

        ~ref_ptr() {
            release(*this);
        }

        operator pointer()       const { return _ptr; }
        operator const_pointer() const { return _ptr; }

        pointer         operator->()       { return _ptr; }
        const_pointer   operator->() const { return _ptr; }

        reference       operator *()       { return*_ptr; }
        const_pointer   operator *() const { return*_ptr; }
        pointer         ptr()              { return _ptr; }
        const_pointer   ptr() const        { return _ptr; }
        reference       ref()              { return*_ptr; }
        const_reference ref() const        { return*_ptr; }
    };

    template<typename R, typename T>
    ref_ptr<R> ref_cast(ref_ptr<T>& rp) {
        T* ptr = rp.ptr();
        return static_cast<R*>(ptr);
    }


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
