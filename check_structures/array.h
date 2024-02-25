//
// Created by Corrado Mio on 25/02/2024.
//
/*
std::array           template < class T, size_t N > class array;

 Member types

    value_type          T
    reference	        value_type&
    const_reference	    const value_type&
    pointer	            value_type*
    const_pointer	    const value_type*
    size_type           size_t
    difference_type	p   trdiff_t

    iterator
    const_iterator
    reverse_iterator        reverse_iterator<iterator>
    const_reverse_iterator  reverse_iterator<const_iterator>

 Member iterators

     begin(),  end(),  cbegin(),  cend(),
    rbegin(), rend(), crbegin(), crend()

 Capacity

    size()
    max_size()
    empty()

 Accessors
    operator[](i)
    at(i)
    front()
    back()
    data()

 Modifiers

    fill(T v)
    swap(array& that)
 */

#ifndef STDX_ARRAY_H
#define STDX_ARRAY_H

#include <cstddef>

#ifndef self
#define self (*this)
# endif

namespace stdx {

    struct info_t {
        size_t refc;    // refcount
        size_t n;       // size
        size_t c;       // capacity

        info_t(size_t c, size_t n): refc(0), c(c), n(n) { }
    };

    template<typename T>
    struct array_t {
    private:

        info_t *info;
        T      *data;

        void add_ref() { info->refc++; }
        void release() { if (0 == --info->refc) {
            delete   info;
            delete[] data;
        }}

        void alloc(size_t c, size_t n) {
            info = new info_t(c, n);
            data = new T[c];
            add_ref();
        }

        void resize(size_t d) {
            size_t c = max_size();
            size_t n = size();

            if (d == c) return;
            if (d  < n) n = d;

            array_t t(*this);   // to avoid de-allocation on release()
            release();          // 't' contains an extra reference of the current array
            alloc(d, n);       // allocate the new capacity (same n)
            init(t);            // copy the content of t
        }

        void init(const T& s) {
            size_t n=size();
            for (int i=0; i<n; ++i) data[i] = s;
        }

        void init(const array_t& a) {
            // THIS array can be shorter than 'a' NEVER longer
            size_t n=size();
            for (int i=0; i<n; ++i) data[i] = a.data[i];
        }

        void assign(const array_t& a) {
            // assign by refcount
            a.add_ref();
            release();
            info = a.info;
            data = a.data();
        }
    public:
        // ------------------------------------------------------------------
        // types
        typedef T                   value_type;
        typedef value_type&         reference;
        typedef const value_type&   const_reference;
        typedef value_type*         pointer;
        typedef const value_type*   const_pointer;
        typedef size_t              size_type;
        typedef ptrdiff_t           difference_type;

    public:
        // ------------------------------------------------------------------
        // constructors

        /// Create an empty array, with max_size 0
        array_t() {
            alloc(0, 0);
        }

        /// Create an array with the specified max_size and size.
        /// If 'n' is -1, size and max_size will be the same
        array_t(int c=0, int n=-1) {
            alloc(c, n<0?c:n);
        }

        /// Create an array by reference or clone
        array_t(const array_t& a, bool clone=false) {
            if (clone) {
                alloc(a.max_size(), a.size());
                init(a);
            }
            else {
                info = a.info;
                data = a.data;
                add_ref();
            }
        }

        ~array_t(){ release(); }

        array_t clone() { return array_t(*this, true); }

        // ------------------------------------------------------------------
        // properties

        [[nodiscard]] size_t size()     const { return info->n; }
        [[nodiscard]] size_t max_size() const { return info->c; }
        [[nodiscard]] bool   empty()    const { return info->n == 0; }

        // ------------------------------------------------------------------
        // accessors

        T& operator[](size_t i)       { return data[i]; }
        T  operator[](size_t i) const { return data[i]; }

        array_t& operator =(const array_t& a) {
            assign(a);
            return *this;
        }

        // ------------------------------------------------------------------
        // operations

        size_t add(const T& e) {
            size_t n = size();
            size_t c = max_size();
            if (n >= c) resize(c == 0 ? 2 : 2*c);

            data[n] = e;
            info->n = n+1;
            return n;
        }

        void clear() {
            info->n = 0;
        }
    };
}

#endif // STDX_ARRAY_H
