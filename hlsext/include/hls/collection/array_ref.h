//
// Created by Corrado Mio on 18/10/2015.
//

#ifndef HLS_COLLECTION_ARRAY_REF_HPP
#define HLS_COLLECTION_ARRAY_REF_HPP

#include <iterator>

namespace hls {
namespace collection {

    /**
     * Stessa interfaccia di std::array e boost::array, MA non gestisce il puntatore
     * all'array.
     *
     * Questo perche' spesso e volentieri, dovendo far convivere array C codice C++, puo'
     * essere utile non dover copiare il vettore in un std::vector<>, oppure non dover creare
     * un std::array<>
     *
     * Nota: NON alloca NE dealloca il vettore!
     *
     * http://www.cplusplus.com/reference/array/array/?kw=array
     */
    template<typename T>
    class array_ref {
    public:
        typedef T                   value_type;
        typedef value_type&         reference;
        typedef const value_type&   const_reference;
        typedef value_type*         pointer;
        typedef const value_type*   const_pointer;
        typedef value_type*         iterator;
        typedef const value_type*   const_iterator;
        typedef std::reverse_iterator<iterator>	        reverse_iterator;
        typedef std::reverse_iterator<const_iterator>   const_reverse_iterator;
        typedef std::size_t         size_type;
        typedef std::ptrdiff_t      difference_type;

    private:
        size_type _size;

        // check range (may be private because it is static)
        void rangecheck(size_type i) const {
            if (i >= _size) {
                std::out_of_range e("array_ref<>: index out of range");
                throw (e);
            }
        }

    public:
        T *elems;

    public:

        size_type size() const { return _size; }
        bool     empty() const { return _size == 0; }
        size_type max_size() { return _size; }

        reference       operator[](size_type i) { return elems[i]; }
        const_reference operator[](size_type i) const { return elems[i]; }

        iterator       begin()       { return elems; }
        iterator         end()       { return elems + _size; }
//        const_iterator cbegin() const { return elems; }
//        const_iterator cend() const { return elems+N; }

        const T *data() const { return elems; }
              T *data()       { return elems; }
           T *c_array()       { return elems; }

        reference at(size_type i) {
            rangecheck(i);
            return elems[i];
        }

        const_reference at(size_type i) const {
            rangecheck(i);
            return elems[i];
        }

//        reference       front()       { return elems[0];   }
//        const_reference front() const { return elems[0];   }
//        reference        back()       { return elems[N-1]; }
//        const_reference  back() const { return elems[N-1]; }

        // assign one value to all elements
//        void assign (const T& value) { fill ( value ); }    // A synonym for fill
//        void fill   (const T& value) { std::fill_n(begin(),size(),value); }

        // swap (note: linear complexity)
        void swap(array_ref<T> &y) {
            assert(size() == y.size());
            for (size_type i = 0; i < _size; ++i)
                std::swap(elems[i], y.elems[i]);
        }

    public:
//        array_ref(const T* a): elems(const_cast<T*>(a)), _size(0) { }
        array_ref(size_type size, const T *a) : elems(const_cast<T *>(a)), _size(size) { }
        array_ref(const array_ref &ar) : elems(ar.elems), _size(ar._size) { }
    };

    template<typename T>
    array_ref<T> aref(const T *p) { return array_ref<T>(p); };

    template<typename T>
    array_ref<T> aref(size_t n, const T *p) { return array_ref<T>(n, p); };

}}

#endif //HLS_COLLECTION_ARRAY_REF_HPP
