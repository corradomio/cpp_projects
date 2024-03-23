//
// Created by Corrado Mio on 20/03/2024.
//
// cuBLAS library uses column-major storage, and 1-based indexing

#ifndef CHECK_CUBLAS_CUBLAS_H
#define CHECK_CUBLAS_CUBLAS_H

#include <exception>
#include <stdexcept>
#include <map>

#undef __cdecl
#include <cuda_runtime.h>
#include <cublas_v2.h>

#ifndef self
#define self (*this)
#endif

#ifndef elif
#define elif else if
#endif


namespace cuda {

    // ----------------------------------------------------------------------
    // cublas_error
    // ----------------------------------------------------------------------

    struct cublas_error : public std::runtime_error {
        static std::map<cublasStatus_t, std::string> CUBLAS_ERRORS;
        // static std::map<cudaError,      std::string> CUDA_ERRORS;
        cublas_error(cublasStatus_t stat): std::runtime_error(CUBLAS_ERRORS[stat]) { }
        cublas_error(cudaError error): std::runtime_error(cudaGetErrorString(error)) { }
    };

    struct bad_dimensions : public std::runtime_error {
        bad_dimensions(): std::runtime_error("Incompatible dimensions") {}
    };

    void check(cublasStatus_t stat);
    void check(cudaError error);

    // ----------------------------------------------------------------------
    // cublas_t
    // ----------------------------------------------------------------------

    class cublas_t {
        struct info_t {
            size_t refc;
            cublasHandle_t handle;
            info_t(): refc(0), handle(nullptr) { }
        };
        mutable info_t *_info;

        void add_ref() const { _info->refc++; }
        void release() const;
        void assign(const cublas_t& cb) {
            _info = cb._info;
        }

    public:
        cublas_t(bool create=true);
        cublas_t(const cublas_t& that) {
            assign(that);
            add_ref();
        }
        ~cublas_t() { release(); }

        cublas_t& create();

        cublas_t& operator =(const cublas_t& that) {
            if (this == &that) { }
            that.add_ref();
            self.release();
            self.assign(that);
            return self;
        }

        operator cublasHandle_t() const { return _info->handle; }

        cublas_t& math_mode(cublasMath_t mode);
        cublasMath_t math_mode() const;
        cublas_t& atomics_mode(cublasAtomicsMode_t mode);
        cublasAtomicsMode_t atomics_mode() const;
    };

    extern thread_local cublas_t context;

    // ----------------------------------------------------------------------
    // real_t
    // device_t
    // layout_t
    // ----------------------------------------------------------------------

    typedef double real_t;

    enum layout_t { ROWS, COLS };
    enum device_t { CPU, GPU };

    // ----------------------------------------------------------------------
    // array_t
    // ----------------------------------------------------------------------

    struct array_t {
        struct info_t {
            size_t refc;
            size_t size;
            device_t dev;
            layout_t lay;

            info_t(size_t n, device_t dev): refc(0), dev(dev), size(n), lay(layout_t::ROWS) { }
        };
        info_t* _info;
        real_t* _data;

        void alloc(size_t n, device_t dev);
        void destroy();
        void add_ref() const;
        void release();
        void assign(const array_t& a);
        void fill(const array_t& a);

        void to_dev(device_t dev);

        [[nodiscard]] size_t  size() const { return self._info->size; }
        [[nodiscard]] real_t* data() const { return self._data;       }
        [[nodiscard]] device_t dev() const { return self._info->dev;  }
    };

    // ----------------------------------------------------------------------
    // vector_t
    // ----------------------------------------------------------------------

    struct vector_t : public array_t {
        using super = array_t;
    public:
        vector_t();
        explicit vector_t(size_t n, device_t dev=CPU);
        vector_t(const vector_t& that):vector_t(that, false) { }
        vector_t(const vector_t& that, bool clone);
        ~vector_t(){ super::release(); }

        vector_t& to(device_t dev) { super::to_dev(dev); return self; }

        vector_t& operator =(const vector_t& that) {
            that.add_ref();
            self.release();
            self.assign(that);
            return self;
        }

        real_t& operator[](size_t i)       { return self._data[i]; }
        real_t  operator[](size_t i) const { return self._data[i]; }
    };

    vector_t zeros(size_t n);
    vector_t  ones(size_t n);
    vector_t range(size_t n);
    vector_t uniform(size_t n, real_t min=0, real_t max=1);

    void print(const vector_t& m);

    // ----------------------------------------------------------------------
    // matrix_t
    // ----------------------------------------------------------------------

    struct matrix_t : public array_t {
        using super = array_t;
        size_t ncols;

        void alloc(size_t rows, size_t cols, device_t dev);
        void assign(const matrix_t&that);
    public:
        matrix_t();
        matrix_t(size_t rows, size_t cols, device_t dev=CPU);
        matrix_t(const matrix_t& m): matrix_t(m, false){ }
        matrix_t(const matrix_t& m, bool clone);

        [[nodiscard]] size_t rows() const { return self.size()/self.ncols; }
        [[nodiscard]] size_t cols() const { return self.ncols; }

        matrix_t& to(device_t dev);// { super::to_dev(dev); return self; }
        matrix_t& layout(layout_t l);
        layout_t layout() const { return self._info->lay; }

        matrix_t& operator =(const matrix_t& that) {
            that.add_ref();
            self.release();
            self.assign(that);
            return self;
        }

        real_t& at(size_t i, size_t j) const {
            if (self.layout() == layout_t::ROWS) {
                size_t n = self.cols();
                return self._data[i*n + j];
            }
            else {
                size_t n = self.rows();
                return self._data[j*n + i];
            }
        }

        real_t& operator[](size_t i) { return self._data[i]; }
        real_t  operator[](size_t i, size_t j) const {
            return self.at(i,j);
        }
        real_t& operator[](size_t i, size_t j) {
            return self.at(i,j);
        }
    };

    matrix_t zeros(size_t rows, size_t cols);
    matrix_t  ones(size_t rows, size_t cols);
    matrix_t range(size_t rows, size_t cols);
    matrix_t identity(size_t rows, size_t cols=-1);
    matrix_t uniform(size_t rows, size_t cols, real_t min=0, real_t max=1);

    void print(const matrix_t& m);

    // ----------------------------------------------------------------------
    // end
    // ----------------------------------------------------------------------

}

#endif //CHECK_CUBLAS_CUBLAS_H
