#include <iostream>
#include <limits>
#include <locale>

// #include "half.hpp"
#include <stdint.h>
#include <stdint-gcc.h>

#include <float.h>
#include <stdfloat>
#include <cstdint>
#include <cfloat>
// #include "bfloat16_t.h"

int main(){
    std::int8_t i8;
    std::int16_t i16;
    std::int32_t i32;
    std::int64_t i64;
    std::float32_t f32;

    return 0;
}

#include "floatx.h"

// FP16    E5M10

typedef union {
    long i;
    float f;
    char c[4];
} value_u;


int main1(){
    numeric::float32_t f = 0;
    f = -1.;
    f = -2.;
    f = -1.5;
    int i = 1<<22;

    numeric::float32_t f1 = -12.5;
    numeric::float32_t f2 = f1;
    numeric::float32_t f3 = f1 + 0.33f;
    numeric::float32_t f4 = 0.33f + f1;
    numeric::float32_t f5 = f2 + f3 + f4;

    long i6 = f5.to_bits();
    numeric::float32_t f6 = numeric::float32_t::from_bits(i6);

    std::cout << f1 << std::endl;
    std::cout << f2 << std::endl;
    std::cout << f3 << std::endl;
    std::cout << f4 << std::endl;
    std::cout << f5 << std::endl;

    std::cout << std::hex << i6 << std::endl;
    std::cout << f6 << std::endl;

    return 0;
}

int main2() {
    value_u v;
    v.f = +.125;  std::cout << std::hex << v.i << " " << v.f << std::endl;
    v.f = +.250;  std::cout << std::hex << v.i << " " << v.f << std::endl;
    v.f = +.500;  std::cout << std::hex << v.i << " " << v.f << std::endl;
    v.f = +1.00;  std::cout << std::hex << v.i << " " << v.f << std::endl;
    v.f = +2.00;  std::cout << std::hex << v.i << " " << v.f << std::endl;

    v.f = -.125;  std::cout << std::hex << v.i << " " << v.f << std::endl;
    v.f = -.250;  std::cout << std::hex << v.i << " " << v.f << std::endl;
    v.f = -.500;  std::cout << std::hex << v.i << " " << v.f << std::endl;
    v.f = -1.00;  std::cout << std::hex << v.i << " " << v.f << std::endl;
    v.f = -2.00;  std::cout << std::hex << v.i << " " << v.f << std::endl;

    return 0;
}

namespace std {

    class binary_num_put
        : public std::num_put<char> {
        template<typename T>
        iter_type common_put(iter_type out, std::ios_base &str, char_type fill,
                             T original, unsigned long long v) const {
            if (str.flags() & std::ios_base::basefield) {
                return this->std::num_put<char>::do_put(out, str, fill, original);
            }
            if (str.flags() & std::ios_base::showbase) {
                *out++ = '0';
                *out++ = str.flags() & std::ios_base::uppercase ? 'B' : 'b';
            }
            unsigned long long mask(1ull << (std::numeric_limits<unsigned long long>::digits - 1));
            while (mask && !(mask & v)) {
                mask >>= 1;
            }
            if (mask) {
                for (; mask; mask >>= 1) {
                    *out++ = v & mask ? '1' : '0';
                }
            } else {
                *out++ = '0';
            }
            return out;
        }

        iter_type do_put(iter_type out, std::ios_base &str, char_type fill, long v) const {
            return common_put(out, str, fill, v, static_cast<unsigned long>(v));
        }

        iter_type do_put(iter_type out, std::ios_base &str, char_type fill, long long v) const {
            return common_put(out, str, fill, v, static_cast<unsigned long long>(v));
        }

        iter_type do_put(iter_type out, std::ios_base &str, char_type fill, unsigned long v) const {
            return common_put(out, str, fill, v, v);
        }

        iter_type do_put(iter_type out, std::ios_base &str, char_type fill, unsigned long long v) const {
            return common_put(out, str, fill, v, v);
        }
    };

    std::ostream &bin(std::ostream &out) {
        auto const &facet = std::use_facet<std::num_get<char>>(out.getloc());
        if (!dynamic_cast<binary_num_put const *>(&facet)) {
            std::locale loc(std::locale(), new binary_num_put);
            out.imbue(loc);
        }
        out.setf(std::ios_base::fmtflags(), std::ios_base::basefield);
        return out;
    }

};

int main3()
{
    std::cout << std::showbase << std::bin << 12345 << " "
              << std::dec << 12345 << "\n";

    return 0;
}

