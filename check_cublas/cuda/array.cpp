//
// Created by Corrado Mio on 22/03/2024.
//
// CUDA Memory Management
//
// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html
//
//  cudaError_t cudaMemcpy ( void* dst, const void* src, size_t count, cudaMemcpyKind kind )
//     Copies data between host and device.
// CUDA memory copy types
// Values
//
//     cudaMemcpyHostToHost = 0     Host -> Host
//     cudaMemcpyHostToDevice = 1   Host -> Device
//     cudaMemcpyDeviceToHost = 2   Device -> Host
//     cudaMemcpyDeviceToDevice = 3 Device -> Device
//     cudaMemcpyDefault = 4
// Direction of the transfer is inferred from the pointer values. Requires unified virtual addressing


#include "cublas.h"

namespace cuda {

    void check(cudaError err) {
        if (err != cudaSuccess)
            throw cublas_error(err);
    }

    // ----------------------------------------------------------------------

    void array_t::add_ref() const {
        _info->refc++;
    }

    void array_t::release() {
        if (0 == --(_info->refc))
            destroy();
    }

    // ----------------------------------------------------------------------

    void array_t::alloc(size_t n, cuda::device_t dev) {
        _info = new info_t(n, dev);
        if (dev == CPU) {
            _data = (real_t *)malloc (n*sizeof (real_t));
        }
        else {
            check(cudaMalloc ((void**)&_data, n*sizeof (real_t)));
        }
    }

    void array_t::destroy() {
        if (_info->dev == CPU) {
            free(_data);
        }
        else {
            check(cudaFree(_data));
        }
    }

    void array_t::assign(const array_t& that) {
        self._info = that._info;
        self._data = that._data;
    }

    void array_t::fill(const array_t& a) {
        size_t n = _info->size;
        if (_info->dev == CPU) {
            // memcpy(_data, a._data, n*sizeof(real_t));
            check(cudaMemcpy(_data, a._data, n*sizeof(real_t), cudaMemcpyHostToHost));
        }
        else {
            check(cudaMemcpy(_data, a._data, n*sizeof(real_t), cudaMemcpyDeviceToDevice));
        }
    }

    // ----------------------------------------------------------------------

    void array_t::to_dev(device_t dev) {
        if (dev == _info->dev)
            return;

        elif (dev == GPU) {
            size_t n = _info->size;
            real_t* data = nullptr;

            check(cudaMalloc ((void**)&data, n*sizeof (real_t)));
            check(cudaMemcpy(data, _data, n*sizeof (real_t), cudaMemcpyHostToDevice));
            free(_data);
            _data = data;
            _info->dev = GPU;
        }
        else {
            size_t n = _info->size;
            real_t* data = nullptr;

            data = (real_t *)(malloc(n * sizeof(real_t)));
            check(cudaMemcpy(data, _data, n*sizeof (real_t), cudaMemcpyDeviceToHost));
            check(cudaFree(_data));
            _data = data;
            _info->dev = CPU;
        }
    }

};