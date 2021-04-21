/* 
 * File:   gnucompat.hpp
 * Author: Corrado Mio
 *
 * Created on April 26, 2015, 11:09 AM
 */

#ifndef STDX_HPP
#define	STDX_HPP

#include <cmath>

namespace hls {
namespace math {

    const float eps = 1.0e-5;

    inline bool iseqz(float x) { return ((x < 0) ? (x >= -eps) : (x <= +eps)); }

    inline bool isltz(float x) { return (x < -eps); }

    inline bool isgtz(float x) { return (x > +eps); }

    inline bool isz(float x) { return iseqz(x - 0); }

    inline bool iso(float x) { return iseqz(x - 1); }

    inline bool iseq(float x, float y) { return iseqz(x - y); }

    inline bool islt(float x, float y) { return isltz(x - y); }

    inline bool isgt(float x, float y) { return isgtz(x - y); }

    inline bool isne(float x, float y) { return !iseqz(x - y); }

    inline bool isle(float x, float y) { return !isgtz(x - y); }

    inline bool isge(float x, float y) { return !isltz(x - y); }

}}


#endif	/* STDX_HPP */

