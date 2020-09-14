
#ifndef NUMCPP_HPP
#define NUMCPP_HPP

#include <w32api/crtdefs.h>

namespace hls {
namespace math {

    class tensor_t {
    public:
        const size_t rank;
        const size_t size[16];
        const size_t length;

    protected:
        void  _init(const size_t* dim);
        size_t  _at(const size_t* index) const;

        tensor_t();
    };

    template<typename T>
    class tensor : public tensor_t {
    public:
        T *data;

        void _create(size_t* dim);
    public:
        tensor():tensor_t() { this->data = new T[1]; };

        tensor* set_size(size_t dim,...) { this->_create(&dim); }
        tensor* set_size(size_t* dim)    { this->_create(dim);  }

        double  at(size_t index, ...) const { return this->data[this->_at(&index)]; };
        double& at(size_t index, ...)       { size_t i = this->_at(&index); return this->data[i]; };
    };

    template<typename T>
    void tensor<T>::_create(size_t* dim)
    {
        delete[] this->data;
        this->_init(dim);
        this->data = new T[this->length];
    }

}};

#endif
