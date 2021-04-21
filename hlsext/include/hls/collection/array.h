//
// Created by Corrado Mio on 29/07/2017.
//

#ifndef DYNARRAY_H
#define DYNARRAY_H

#include <cassert>
#include <iterator>

namespace hls {
namespace collection {

    /**
     * Array con la semantica del reference count.
     * L'array non viene clonato, come in std::vector, ma passato come
     * puntatore + reference count
     *
     * http://www.cplusplus.com/reference/array/array/?kw=array
     */

    template<typename _Tp>
    class array {
    public:
        typedef _Tp                 value_type;
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
        struct info {
            size_type refcount;     //
            size_type size;         // dimensione del puntatore

            info(size_t sz): refcount(0), size(sz){ }
            ~info(){
                assert(refcount == 0);
                refcount = size = 0;
            }
        } *_info;
        pointer _data;

        void add_ref() { ++_info->refcount; }
        void release() {
            if(--_info->refcount == 0) {
                delete _info;
                delete[] _data;
            };
        }

    public:

        // ------------------------------------------------------------------
        //
        // ------------------------------------------------------------------

        array():_info(new info(0)), _data(nullptr){ add_ref(); }

        explicit array(size_type sz):_info(new info(sz)), _data(new value_type[sz]) { add_ref(); }

        array(size_type sz, pointer dt):_info(new info(sz)), _data(dt){ add_ref(); }

        array(const array& ary):_info(ary._info), _data(ary._data) { add_ref(); }

        ~array() { release(); }

        // ------------------------------------------------------------------
        //
        // ------------------------------------------------------------------

        size_type last() const { return size() - 1; }

        pointer data() const { return _data; }

        // ------------------------------------------------------------------
        //
        // ------------------------------------------------------------------

        array& operator =(const array& ary) {
            ary.add_ref();
            release();
            _info = ary._info;
            _data = ary._data;
            return *this;
        }

        // ------------------------------------------------------------------

        reference       at(size_type pos)       { return _data[pos]; }
        const_reference at(size_type pos) const { return _data[pos]; }

        reference       operator[](size_type index)       { return _data[index]; }
        const_reference operator[](size_type index) const { return _data[index]; }

        reference       front()       { return _data[0]; }
        const_reference front() const { return _data[0]; }
        reference        back()       { return _data[last()]; }
        const_reference  back() const { return _data[last()]; }

        // ------------------------------------------------------------------

        iterator         begin()       { return _data + 0; }
        iterator           end()       { return _data + size(); }
        const_iterator cbeging() const { return _data + 0; }
        const_iterator    cend() const { return _data + size(); }

        iterator        rbegin() { return _data + last(); }
        iterator          rend() { return _data - 1; }
        const_iterator crbegin() { return _data + last(); }
        const_iterator   crend() { return _data - 1; }

        // ------------------------------------------------------------------

        bool empty() const { return _info->size == 0; }

        size_type size()     const { return _info->size; }
        size_type max_size() const { return _info->size; }

        // ------------------------------------------------------------------

        void fill(const_reference value) {
            for(iterator it=begin(); it<end(); ++it)
                *it = value;
        }

        void swap(array& other) {
            std::swap(_data, other._data);
            std::swap(_info, other._info);
        }

    };
}}

#endif //DYNARRAY_H
