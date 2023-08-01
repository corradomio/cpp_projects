//
// Created by Corrado Mio on 01/08/2023.
//

#ifndef CHECK_MKL_MAIN_H
#define CHECK_MKL_MAIN_H

namespace c10 {

    enum class DeviceType : int8_t {
        CPU = 0,
        CUDA = 1, // CUDA.
        MKLDNN = 2, // Reserved for explicit MKLDNN
        OPENGL = 3, // OpenGL
    };

}

#endif //CHECK_MKL_MAIN_H
