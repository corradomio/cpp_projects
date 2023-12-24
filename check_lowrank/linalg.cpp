//
// Created by Corrado Mio on 16/12/2023.
//
#include <tuple>
#include <random>
#include <exception>

#include "linalg.h"

typedef unsigned char byte;

using namespace linalg;

// --------------------------------------------------------------------------
// dim_t
// --------------------------------------------------------------------------

size_t _avail(size_t r) {
    size_t r4 = r%4;
    return (r4 == 0) ? r : r + (4 - r4);
}

dim_t:: dim_t() {
    rank = 0;
    avail = _avail(rank);
    dim = new size_t[avail];
}

dim_t::dim_t(std::initializer_list<size_t> dims) {
    size_t i = 0;
    rank = dims.size();
    avail = _avail(rank);
    dim = new size_t[avail];
    for(auto it=dims.begin(); it<dims.end(); ++it)
        dim[i++] = *it;
}

dim_t::dim_t(const dim_t& d): rank(d.rank), dim(new size_t[d.rank]) {
    rank = d.rank;
    avail = d.avail;
    dim = new size_t[avail];
    memcpy(dim, d.dim, rank*sizeof(size_t));
}

dim_t::~dim_t(){
    delete[] dim;
    dim = nullptr;
    rank = 0;
}

size_t dim_t::size() const {
    size_t s = 1;
    for (int i=0; i<avail; ++i) s *= dim[i];
    return s;
}

dim_t& dim_t::operator=(const dim_t& d) {
    delete[] dim;
    rank = d.rank;
    avail = d.avail;
    dim = new size_t[avail];
    memcpy(dim, d.dim, avail*sizeof(size_t));
    return *this;
}

// --------------------------------------------------------------------------
// tensor_t
// --------------------------------------------------------------------------

tensor_t::tensor_t()
: dims() {
    data = new float[1];
}

tensor_t::tensor_t(const std::initializer_list<size_t>& dims)
: dims(dims) {
    data = new float[dims.size()];
}

tensor_t::tensor_t(const dim_t& dims)
    : dims(dims) {
    data = new float[dims.size()];
}

tensor_t::~tensor_t() {
    delete[] data;
    data = nullptr;
}

// --------------------------------------------------------------------------
// constructors
// --------------------------------------------------------------------------

tensor_t* linalg::newt(const std::initializer_list<size_t>& dims) {
    return new tensor_t(dims);
}

tensor_t* linalg::zeros(const std::initializer_list<size_t>& dims) {
    tensor_t* t = linalg::newt(dims);
    size_t n = t->dims.size();
    float* data = t->data;
    for(size_t i=0; i<n; ++i)
        data[i] = 0;
    return t;
}

tensor_t*  linalg::ones(const std::initializer_list<size_t>& dims) {
    tensor_t* t = linalg::newt(dims);
    size_t n = t->dims.size();
    float* data = t->data;
    for(size_t i=0; i<n; ++i)
        data[i] = 1;
    return t;
}

tensor_t*  linalg::rand(const std::initializer_list<size_t>& dims) {
    tensor_t* t = linalg::newt(dims);
    size_t n = t->dims.size();
    float rmax = static_cast <float> (RAND_MAX);
    float* data = t->data;
    for(size_t i=0; i<n; ++i)
        data[i] = static_cast <float> (::rand()) / rmax;
    return t;
}

tensor_t* linalg::newt(tensor_t* t) {
    tensor_t* c = new tensor_t(t->dims);
    memcpy(c->data, t->data, t->dims.size()*sizeof(float));
    return c;
}

tensor_t* newt(float f) {
    tensor_t* c = new tensor_t();
    c->data[0] = f;
    return c;
}



// tensor_t* linalg::newt(size_t dim0, size_t dim1) {
//     tensor_t* t;
//     size_t size;
//
//     if (dim0 <= 0) {
//         size = 1;
//         t = (tensor_t*)(new ::byte[sizeof(tensor_t) + size*sizeof(float)]);
//         t->dims.rank = 0;
//         t->dims.dim[0] = 0;
//         t->dims.dim[1] = 0;
//     }
//     else if (dim1 <= 0) {
//         size = dim0;
//         t = (tensor_t*)(new ::byte[sizeof(tensor_t) + size*sizeof(float)]);
//         t->dims.rank = 1;
//         t->dims.dim[0] = dim0;
//         t->dims.dim[1] = 0;
//     }
//     else {
//         size = dim0*dim1;
//         t = (tensor_t*)(new ::byte[sizeof(tensor_t) + size*sizeof(float)]);
//         t->dims.rank = 2;
//         t->dims.dim[0] = dim0;
//         t->dims.dim[1] = dim1;
//     }
//     t->refc=0;
//     return t;
// }
//
//
// tensor_t* linalg::newt(float v) {
//     tensor_t* t = newt(0, 0);
//     t->data[0] = v;
//     return t;
// }
//
//
// tensor_t* linalg::newt(tensor_t* t) {
//     tensor_t* u = newt(t->dims.dim[0], t->dims.dim[1]);
//     u->dims.rank = t->dims.rank;
//     return u;
// }
//
//
// tensor_t* linalg::zeros(size_t dim0, size_t dim1) {
//     tensor_t* t = linalg::newt(dim0, dim1);
//     size_t n = t->dims.size();
//     for (int i=0; i<=n; ++i)
//         t->data[i] = 0.;
//     return t;
// }
//
// tensor_t* linalg::ones(size_t dim0, size_t dim1) {
//     tensor_t* t = linalg::newt(dim0, dim1);
//     size_t n = t->dims.size();
//     for (int i=0; i<=n; ++i)
//         t->data[i] = 1.;
//     return t;
// }
//
// tensor_t* linalg::rand(size_t dim0, size_t dim1) {
//     tensor_t* t = linalg::newt(dim0, dim1);
//     size_t n = t->dims.size();
//     float rmax = static_cast <float> (RAND_MAX);
//     for (int i=0; i<=n; ++i)
//         t->data[i] = static_cast <float> (::rand()) / rmax;
//     return t;
// }
//
// // --------------------------------------------------------------------------
//
// void check_elw_dims(const tensor& t1, const tensor& t2) {
//     if (t1.rank() != t2.rank())
//         throw std::exception();
//     else if (t1.rank() == 0 && t2.rank() == 0)
//         return;
//     else if (t1.rank() == 1 && t2.rank() == 1) {
//         if (t1.dim(0) != t2.dim(0))
//             throw std::exception();
//     }
//     else if (t1.rank() == 2 && t2.rank() == 2) {
//         if (t1.dim(0) != t2.dim(0))
//             throw std::exception();
//         if (t1.dim(1) != t2.dim(1))
//             throw std::exception();
//     }
// }
//
// void check_dot_dims(const tensor& t1, const tensor& t2) {
//     if (t1.rank() == 0 || t2.rank() == 0)
//         return;
//     else if (t1.rank() == 1 && t2.rank() == 1) {
//         if (t1.dim(0) != t2.dim(0))
//             throw std::exception();
//     }
//     else if (t1.rank() == 1 && t2.rank() == 2) {
//         if (t1.dim(0) != t2.dim(0))
//             throw std::exception();
//     }
//     else if (t1.rank() == 2 && t2.rank() == 1) {
//         if (t1.dim(1) != t2.dim(0))
//             throw std::exception();
//     }
//     if (t1.rank() == 2 && t2.rank() == 2) {
//         if (t1.dim(1) != t2.dim(0))
//             throw std::exception();
//     }
// }
//
// tensor tensor::dot(tensor t) {
//     check_dot_dims(*this, t);
//
//     if (rank() == 0 && t.rank() == 0)
//         return linalg::newt(at()*t.at());
//
//     if (rank() == 0) {
//         float s = at();
//         tensor r = newt(t.ptr());
//         for(int i=0; i<t.size(); ++i)
//             r[i] = s*t[i];
//         return r;
//     }
//     if (t.rank() == 0) {
//         float s = t.at();
//         tensor r = newt(ptr());
//         for(int i=0; i<size(); ++i)
//             r[i] = s*at(i);
//         return r;
//     }
//     if (rank() == 1 && t.rank() == 1) {
//         float s = 0;
//         for(int i=0; i<size(); ++i)
//             s += at(i)*t.at(i);
//         return newt(s);
//     }
//     if (rank() == 1) {
//         tensor r = newt(t.dim(1));
//     }
//
// }
