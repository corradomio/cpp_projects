/* 
 * File:   gnucompat.hpp
 * Author: Corrado Mio
 *
 * Created on April 26, 2015, 11:09 AM
 */

#ifndef STDX_HPP
#define	STDX_HPP

#include <cmath>

namespace stdx {
namespace math {

    const double eps = 1.0e-8;
    // x < 0
    inline bool isltz(double x) { return (x < -eps); }
    // x > 0
    inline bool isgtz(double x) { return (x > +eps); }
    // x = 0
    inline bool isz(double x) { return (-eps <= x) && (x <= +eps); }
    // x = y
    inline bool iseq(double x, double y) { return isz(x - y); }
    // x < y
    inline bool islt(double x, double y) { return isltz(x - y); }
    // x > y
    inline bool isgt(double x, double y) { return isgtz(x - y); }
    // x != y
    inline bool isne(double x, double y) { return !isz(x - y); }
    // x <= y
    inline bool isle(double x, double y) { return !isgtz(x - y); }
    // x >= y
    inline bool isge(double x, double y) { return !isltz(x - y); }

    // x in [min,max]
    inline bool isin(double x, double min, double max) {
        return (min-eps) <= x && x <= (max + eps);
    }

    inline bool isbetween(double x, double min, double max) {
        return min <= x && x <= max;
    }
}}


#endif	/* STDX_HPP */

