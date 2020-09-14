//
// Created by Corrado Mio on 28/02/2016.
//

/*

#define MAXFLOAT	3.40282347e+38F
#define _M_LN2      0.69314718055994530941

#define M_E	        2.7182818284590452354
#define M_LOG2E		1.4426950408889634074
#define M_LOG10E	0.43429448190325182765
#define M_LN2		_M_LN2
#define M_LN10		2.30258509299404568402
#define M_PI		3.14159265358979323846
#define M_PI_2		1.57079632679489661923
#define M_PI_4		0.78539816339744830962
#define M_1_PI		0.31830988618379067154
#define M_2_PI		0.63661977236758134308
#define M_2_SQRTPI	1.12837916709551257390
#define M_SQRT2		1.41421356237309504880
#define M_SQRT1_2	0.70710678118654752440

#define M_TWOPI     (M_PI * 2.0)
#define M_3PI_4		2.3561944901923448370E0
#define M_SQRTPI    1.77245385090551602792981
#define M_LN2LO     1.9082149292705877000E-10
#define M_LN2HI     6.9314718036912381649E-1
#define M_SQRT3	    1.73205080756887719000
#define M_IVLN10    0.43429448190325182765   // 1 / log(10)
#define M_LOG2_E    _M_LN2
#define M_INVLN2    1.4426950408889633870E0  // 1 / log(2)

 */

#include <math.h>
#include <float.h>

#ifndef TBBTEST_MATHCONST_HPP
#define TBBTEST_MATHCONST_HPP

namespace std {
namespace math {

    const float  minfloat  = 1.175494351e-38F;
    const float  maxfloat  = 3.402823466e+38F;

    const double mindouble = 2.2250738585072014e-308;
    const double maxdouble = 1.7976931348623157e+308;

    const double e      = 2.7182818284590452354;
    const double log2e  = 1.4426950408889634074;
    const double log10e = 0.43429448190325182765;
    const double ln2    = 0.693147180559945309417;
    const double ln10   = 2.30258509299404568402;

    const double pi     = 3.14159265358979323846;   // PI
    const double pi2    = 1.57079632679489661923;   // PI/2
    const double pi4    = 0.78539816339744830962;   // PI/4
    const double invpi  = 0.31830988618379067154;   // 1/PI

    const double twopi  = (2.0*pi);                 // 2*pi
    const double sqrtpi = 1.77245385090551602792981;// sqrt(PI)
    const double sqrt3  = 1.73205080756887719000;   // sqrt(3)

    const double invln10 = 0.43429448190325182765;  // 1/ln(10)
    const double invln2  = 1.44269504088896338700;  // 1/ln(2)

}};

#endif //TBBTEST_MATHCONST_HPP
