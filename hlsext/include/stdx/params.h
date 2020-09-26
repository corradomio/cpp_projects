//
// Created by Corrado Mio on 26/09/2020.
//

#ifndef CHECK_HLSEXT_PARAMS_H
#define CHECK_HLSEXT_PARAMS_H

#include <tuple>
#include <vector>

namespace stdx {

    template<typename T>
    std::vector<std::tuple<T>> make_params(std::vector<T>& p) {  return p; }

    template<typename T, typename ... Rest>
    std::vector<std::tuple<T, Rest>> make_params(std::vector<T>& p, Rest ... rest) {
        std::vector<std::tuple<Rest>> rparams = make_params<Rest>(rest);
        return p;
    }

}

#endif //CHECK_HLSEXT_PARAMS_H
