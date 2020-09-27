//
// Created by Corrado Mio on 26/09/2020.
//

#ifndef CHECK_HLSEXT_PARAMS_H
#define CHECK_HLSEXT_PARAMS_H

#include <tuple>
#include <vector>

namespace stdx {
    template<typename T>
    T adder(T v) {
        return v;
    }

    template<typename T, typename... Args>
    T adder(T first, Args... args) {
        return first + adder(args...);
    }

    template<typename T>
    std::vector<std::tuple<T>> make_params(const std::vector<T>& plist) {
        std::vector<std::tuple<T>> params;
        for(auto p : plist)
            params.push_back(std::make_tuple(p));
        return params;
    }

    template<typename T, typename ... Rest>
    std::vector<std::tuple<T, Rest ...>> make_params(const std::vector<T>& plist, const std::vector<Rest>&... rest) {
        std::vector<std::tuple<Rest...>> rparams = make_params<Rest...>(rest...);
        std::vector<std::tuple<T, Rest...>> params;

        for (auto p : plist) {
            std::tuple<T> tp = std::make_tuple(p);
            for (auto tr : rparams) {
                std::tuple<T, Rest...> tpr = std::tuple_cat(tp, tr);
                params.push_back(tpr);
            }
        }

        return params;
    }

}

#endif //CHECK_HLSEXT_PARAMS_H
