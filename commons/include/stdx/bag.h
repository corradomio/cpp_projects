//
// Created by Corrado Mio on 28/07/2024.
//

#ifndef STDX_BAG_H
#define STDX_BAG_H

#include <language.h>
#include <map>

namespace stdx {

    template<typename T>
    class bag : public std::map<T, size_t> {
        using super = std::map<T, size_t>;
    public:
        bag() { }

        size_t insert(const T& item) {
            size_t count;
            if (self.find(item) == self.end()) {
                count = 1;
                self.emplace(std::pair(item, count));
            }
            else {
                count = self.at(item) + 1;
                self.at(item) = count;
            }
            return count;
        }
    };
}

#endif //STDX_BAG_H
