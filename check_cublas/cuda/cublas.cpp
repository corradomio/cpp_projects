//
// Created by Corrado Mio on 20/03/2024.
//

#include "cublas.h"

namespace cuda {

    thread_local cublas_t context{true};

    void check(cublasStatus_t stat) {
        if (stat != CUBLAS_STATUS_SUCCESS)
            throw cublas_error(stat);
    }

    // ----------------------------------------------------------------------

    void cublas_t::release() const {
        if (0 == --(_info->refc)) {
            if (_info->handle != nullptr)
                cublasDestroy(_info->handle);
            delete _info;
        }
    }

    cublas_t::cublas_t(bool create): _info(new info_t) {
        if (create) self.create();
        add_ref();
    }

    cublas_t& cublas_t::create() {
        if (_info->handle == nullptr)
            check(cublasCreate(&(_info->handle)));
        return self;
    }

    // ----------------------------------------------------------------------

    cublas_t& cublas_t::math_mode(cublasMath_t mode) {
        check(cublasSetMathMode(self, mode));
        return self;
    }

    cublasMath_t cublas_t::math_mode() const {
        cublasMath_t mode;
        check(cublasGetMathMode(self, &mode));
        return mode;
    }

    cublas_t& cublas_t::atomics_mode(cublasAtomicsMode_t mode) {
        check(cublasSetAtomicsMode(self, mode));
        return self;
    }

    cublasAtomicsMode_t cublas_t::atomics_mode() const {
        cublasAtomicsMode_t mode;
        check(cublasGetAtomicsMode(self, &mode));
        return mode;
    }

    // ----------------------------------------------------------------------

}