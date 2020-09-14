#include <stdio.h>
#include <cstring>
#include "../../../include/hls/math/float4.hpp"

using namespace hls::math;


#define _asm_ __asm__ volatile

// ==========================================================================
// Support
// ==========================================================================

static float eps = 1.0e-5;

static inline bool isz(float x) { return x < 0 ? x >= -eps : x <= +eps; }
static inline bool iseq(float x, float y) { return isz(x-y); }
static inline float sqf(float x) { return x*x; }


// ==========================================================================
// Constants
// ==========================================================================

float4 f4::origin(0,0,0);

float4 f4::x_axis(1,0,0);
float4 f4::y_axis(0,1,0);
float4 f4::z_axis(0,0,1);

float4 f4::neg_x_axis(-1,0,0);
float4 f4::neg_y_axis(0,-1,0);
float4 f4::neg_z_axis(0,0,-1);


// ==========================================================================
// SSE implementations
// ==========================================================================

//static float sqrtps(const float x) {
//    float ret = 0;
//    _asm_ (
//        "movaps %[x],   %%xmm0;"
//        "sqrtss %%xmm0, %%xmm0;"
//        "movss  %%xmm0, %[ret];"
//        : [ret] "=m" (ret)
//        : [x] "m" (x)
//    );
//    return ret;
//}

static float dotps(const __m128& a, const __m128& b, bool squared=false) {
   
    // 4.7 ns -O1
    __m128 t128 = _mm_mul_ps(a, b);
    t128 = _mm_add_ps(t128, _mm_shuffle_ps(t128, t128, _MM_SHUFFLE(3,3,3,2)));
    t128 = _mm_add_ss(t128, _mm_shuffle_ps(t128, t128, _MM_SHUFFLE(3,3,3,1)));
    if (squared)
        t128 = _mm_sqrt_ss(t128);
    return t128[0];
    
    // 7.9 ns
//    float ret = 0;
//    _asm_ (
//        "movaps %[a],   %%xmm0;"
//        "mulps  %[b],   %%xmm0;"
//        "movaps %%xmm0, %%xmm1;"
//        "shufps $0xFE,  %%xmm1, %%xmm1;"        // 3 3 3 2
//        "addps  %%xmm1, %%xmm0;"
//        "movaps %%xmm0, %%xmm1;"
//        "shufps $0xF9,  %%xmm1, %%xmm1;"        // 3 3 3 1
//        "addps  %%xmm1, %%xmm0;"
//        //"movss  %%xmm0, %[ret];"
//        : [ret] "=m" (ret)
//        :   [a]  "m" (a), [b] "m" (b)
//    );
//        
//    if (!squared) 
//    {
//        _asm_ (
//            "movss  %%xmm0, %[ret];"
//            : [ret] "=m" (ret)
//        );
//    }
//    else 
//    {
//        _asm_ (
//            "sqrtss %%xmm0, %%xmm0;"
//            "movss  %%xmm0, %[ret];"
//            : [ret] "=m" (ret)
//        );
//    }
//    return ret;
}

//static float normps(const __m128& a) {
//    float ret = 0;
//    __asm__ (
//        "movaps %[a],   %%xmm0;"
//        "mulps  %%xmm0, %%xmm0;"
//        "movaps %%xmm0, %%xmm1;"
//        "shufps $0xFE,  %%xmm1, %%xmm1;"        // 3 3 3 2
//        "addps  %%xmm1, %%xmm0;"
//        "movaps %%xmm0, %%xmm1;"
//        "shufps $0xF9,  %%xmm1, %%xmm1;"        // 3 3 3 1
//        "addps  %%xmm1, %%xmm0;"
//        "sqrtss %%xmm0, %%xmm0;"
//        "movss  %%xmm0, %[ret];"
//        : [ret] "=m" (ret)
//        :   [a]  "m" (a)
//    );
//    return ret;
//}

static __m128  epsps = (__m128){ eps, eps, eps, eps };
static __m128 signps = (__m128)(__v4su){ 0x80000000, 0x80000000, 0x80000000, 0x80000000 };


//__m128 hls::absps(const __m128& a) {
//    __m128 ret;
//    __asm__ (
//        "movaps %[a],    %%xmm2;"
//        "movaps %[sign], %%xmm0;"
//        "andnps %%xmm2,  %%xmm0;"
//        "movaps %%xmm0,  %[ret]"
//        : [ret] "=m" (ret)
//        :   [a]  "m" (a), [sign] "m" (signps), [eps] "m" (epsps)
//    );
//    return ret;
//}

//bool iszps(const __m128& a) {
//    __m128 ret;
//    _asm_ (
//        // clear bit r[31]
//        "movaps %[sign], %%xmm0;"
//        "andnps %[a],    %%xmm0;"
//    
//        // check if eps < r  (opposite of 'r <= eps' to have AL zero if true)
//        "movaps %[eps],  %%xmm1;"
//        "cmpps  $1, %%xmm0, %%xmm1;"
//        "movaps %%xmm1, %[ret]"
//        : [ret] "=m" (ret)
//        :   [a]  "m" (a), 
//         [sign] "m" (signps), [eps] "m" (epsps)
//    );
//    return !(ret[0] || ret[1] || ret[2] || ret[3]);
//}

//bool iseqps(const __m128& a, const __m128& b) {
//    __m128 ret;
//    _asm_ (
//        "movaps %[sign], %%xmm0;"
//
//        // r = a - b
//        "movaps %[a],    %%xmm1;"
//        "subps  %[b],    %%xmm1;"
//    
//        // clear bit r[31]
//        "andnps %%xmm1,  %%xmm0;"
//    
//        // check if eps < r  (opposite of 'r <= eps' to have AL zero if true)
//        "movaps %[eps],  %%xmm1;"
//        "cmpps  $1, %%xmm0, %%xmm1;"
//        "movaps %%xmm1, %[ret]"
//        : [ret] "=m" (ret)
//        :   [a] "m" (a), [b] "m" (b), 
//         [sign] "m" (signps), [eps] "m" (epsps)
//    );
//    return !(ret[0] || ret[1] || ret[2] || ret[3]);
//}


// ==========================================================================
// float4
// ==========================================================================

// -- predicates

bool float4::zero() const {
    // 5.5 ns
    //return isz(x) && isz(y) && isz(z);
    
    // 9.9 ns
    __m128 c128 = _mm_cmple_ps(epsps, _mm_andnot_ps(signps, m128));
    return !(c128[0] || c128[1] || c128[2]);
    
    // 5.1 ns
    //return iszps(m128);
}

bool float4::equal(const float4& f) const {
    // 19.8 ns
    //return iseq(x, f.x) && iseq(y, f.y) && iseq(z, f.z);

    // 10.5 ns
    __m128 c128 = _mm_cmple_ps(epsps, _mm_andnot_ps(signps, _mm_sub_ps(m128, f.m128)));
    return !(c128[0] || c128[1] || c128[2]);
    
    // 11.5 ns
    //return iseqps(m128, f.m128);
}

// -- scalar

float float4::dot(const float4& f) const {
    // 5.1 ns
    //return x*f.x + y*f.y + z*f.z;
    
    // 10.7 ns
    //__m128 t128 = _mm_mul_ps(m128, f.m128);
    //t128 = _mm_add_ps(t128, _mm_shuffle_ps(t128, t128, _MM_SHUFFLE(3,3,3,2)));
    //t128 = _mm_add_ss(t128, _mm_shuffle_ps(t128, t128, _MM_SHUFFLE(3,3,3,1)));
    //return t128[0];
    
    // 7.2 ns (asm))
    return dotps(m128, f.m128);
    
    // 5.1 ns
    //float ret;
    //__asm__ (
    //    "movaps %[a],   %%xmm0;"
    //    "mulps  %[b],   %%xmm0;"
    //    "movaps %%xmm0, %%xmm1;"
    //    "shufps $0xFE,  %%xmm1, %%xmm1;"        // 3 3 3 2
    //    "addps  %%xmm1, %%xmm0;"
    //    "movaps %%xmm0, %%xmm1;"
    //    "shufps $0xF9,  %%xmm0, %%xmm1;"        // 3 3 3 1
    //    "addps  %%xmm1, %%xmm0;"
    //    "movss  %%xmm0, %[ret];"
    //    : [ret] "=m" (ret)
    //    : [a] "m" (m128), [b] "m" (f.m128)
    //);
    //return ret;
}

float float4::norm() const {
    // 41.7 ns
    //return sqrtf(sqf(x) + sqf(y) + sqf(z));
    
    // 38.8 ns, dot/asm
    //return sqrtf(dot(*this));
    
    // 39.2 ns  -O0
    // 5.4 ns   -O1
    return sqrtf(dotps(m128, m128));

    // 18.6 ns
    //return sqrtps(dotps(m128, m128));
    
    // 8.2 ns
    //return dotps(m128, m128, true);
    
    // 5.1 ns
    //float ret;
    //__asm__ (
    //    "movaps %[a],   %%xmm0;"
    //    "mulps  %%xmm0, %%xmm0;"
    //    "movaps %%xmm0, %%xmm1;"
    //    "shufps $0x1E,  %%xmm1, %%xmm1;"        // 0 1 2 3
    //    "addps  %%xmm1, %%xmm0;"
    //    "movaps %%xmm0, %%xmm1;"
    //    "shufps $0x39,  %%xmm0, %%xmm1;"        // 0 3 2 1
    //    "addps  %%xmm1, %%xmm0;"
    //    "sqrtss %%xmm0, %%xmm0;"
    //    "movss  %%xmm0, %[ret];"
    //    : [ret] "=m" (ret)
    //    : [a] "m" (m128)
    //);
    //return ret;
}

// -- vector

//float4 unit(const float4& f) {
//    float m = f.norm();
//    return f*(isz(m) ? 0.f : (1/m));
//}


float4 float4::cross(const float4& f) const {
    // y1*z2  z1*x2  x1*y2
    // z1*y2  x1*z2  y1*x2  
    
    // 11.8 ns
    //return float4(
    //        y*f.z - z*f.y,
    //        z*f.x - x*f.z,
    //        x*f.y - y*f.x);
    
    // 11.8 ns
    return _mm_sub_ps(
        _mm_mul_ps(_mm_shuffle_ps(m128, m128, _MM_SHUFFLE(3, 0, 2, 1)), _mm_shuffle_ps(f.m128, f.m128, _MM_SHUFFLE(3, 1, 0, 2))), 
        _mm_mul_ps(_mm_shuffle_ps(m128, m128, _MM_SHUFFLE(3, 1, 0, 2)), _mm_shuffle_ps(f.m128, f.m128, _MM_SHUFFLE(3, 0, 2, 1)))
    );
    
    // 7.6 ns
    //__m128 r128;
    //__asm__ (
    //    "movaps %[a], %%xmm3;"              //  3  2  1  0
    //    "movaps %[b], %%xmm4;"
    //
    //    "movaps %%xmm3, %%xmm0;"
    //    "movaps %%xmm4, %%xmm1;"
    //    "shufps $0xC9,  %%xmm0, %%xmm0;"    // t1 x1 z1 y1
    //    "shufps $0xD2,  %%xmm1, %%xmm1;"    // t2 y2 x2 z2
    //    "mulps  %%xmm1, %%xmm0;"
    //
    //    "movaps %%xmm3, %%xmm1;"
    //    "movaps %%xmm4, %%xmm2;"
    //    "shufps $0xD2,  %%xmm1, %%xmm1;"    // t1 y1 x1 z1
    //    "shufps $0xC9,  %%xmm2, %%xmm2;"    // t2 x2 z2 y2   
    //    "mulps  %%xmm2, %%xmm1;"
    //
    //    "subps  %%xmm1, %%xmm0;"
    //    "movaps %%xmm0, %[ret];"
    //    : [ret] "=m" (r128)
    //    : [a] "m" (m128), [b] "m" (f.m128)
    //);
    //return r128;
}

        
float4 float4::ortho() const {
    return unit(*this);
}

float4 float4::ortho(const float4& v) const {
    float4 t(*this);
    t -= t.dot(v)*v;
    return unit(t);
}

float4 float4::ortho(const float4& u, const float4& v) const {
    float4 t(*this);
    t -= t.dot(u)*u;
    t -= t.dot(v)*v;
    return unit(t);
}

// -- str

std::string float4::str() const {
    char buf[64];
    sprintf(buf, "[%f, %f, %f]", x, y, z);
    //sprintf(buf, "[%f, %f, %f]", m128[0], m128[1], m128[2]);
    return buf;
}

// ==========================================================================
// float4x4
// ==========================================================================

// -- matrix constructors

float4x4::float4x4() {
    identity();
}

float4x4::float4x4(const float* t) {
    size_t i,j,k=0;
    for(i=0; i<4; ++i)
        for(j=0; j<4; ++j,++k)
            m[i][j] = t[k];
}

float4x4::float4x4(const float4x4& t, transposed tr) {
    size_t i,j,k=0;
    for(i=0; i<4; ++i)
        for(j=0; j<4; ++j,++k)
            at(j,i) = t.at(i,j);
}



float4x4& float4x4::zero() {
    memset(&v512, 0, sizeof(v512));
    return *this;
}

float4x4& float4x4::identity() {
    memset(&v512, 0, sizeof(v512));
    at(0,0) = at(1,1) = at(2,2) = at(3,3) = 1;
    return *this;
}

float4x4& float4x4::translation(float x, float y, float z) {
    identity();
    at(0,3) = x;
    at(1,3) = y;
    at(2,3) = z;
    return *this;
}

float4x4& float4x4::rotation(float x, float y, float z, float c, float s) {
    float m = sqrtf(sqf(x) + sqf(y) + sqf(z));
    float q = sqrtf(sqf(c) + sqf(s));

    identity();

    if (isz(m) || isz(q)) 
        return *this;

    // normalize (x,y,z) and (c,s)
    {
        m = 1/m;
        x *= m;
        y *= m;
        z *= m;

        q = 1/q;
        c *= q;
        s *= q;
    }

    if (iseq(c,1))
    {
        /* none */
    }
    else if (iseq(c, -1))
    {
        if (iseq(z, 1))
        {
            at(0,0) = -1;
            at(2,2) = -1;
        }
        else if (iseq(z, -1))
        {
            at(0,0) = -1;
            at(2,2) = -1;
        }
        else
        {
            float x2 = sqf(x);
            float y2 = sqf(y);
            float z2 = sqf(z);
            float xy = x*y;
            float s = 1/(1 - z2);

            at(0,0) = -(x2 - y2)*s;
            at(0,1) = -2*x*y*s;
            at(1,0) = -2*x*y*s;
            at(1,1) = (x2 - y2)*s;
            at(2,2) = -(1 - z2);
        }
    }
    else if (iseq(x, 1))
    {
        at(1,1) =  c;
        at(1,2) = -s;
        at(2,1) =  s;
        at(2,2) =  c;
    }
    else if (iseq(x,-1))
    {
        at(1,1) =  c;
        at(1,2) =  s;
        at(2,1) = -s;
        at(2,2) =  c;
    }
    else if (iseq(y, 1))
    {
        at(0,0) =  c;
        at(0,2) =  s;
        at(2,0) = -s;
        at(2,2) =  c;
    }
    else if (iseq(y,-1))
    {
        at(0,0) =  c;
        at(0,2) = -s;
        at(2,0) =  s;
        at(2,2) =  c;
    }
    else if (iseq(z, 1))
    {
        at(0,0) =  c;
        at(0,1) = -s;
        at(1,0) =  s;
        at(1,1) =  c;
    }
    else if (iseq(z,-1))
    {
        at(0,0) =  c;
        at(0,1) =  s;
        at(1,0) = -s;
        at(1,1) =  c;
    }
    else
    {
        float x2 = sqf(x);
        float y2 = sqf(y);
        float z2 = sqf(z);
        float xy = x*y;
        float yz = y*z;
        float xz = x*z;

        at(0,0) = c + x2*(1 - c);
        at(0,1) = xy*(1 - c) - z*s;
        at(0,2) = xz*(1 - c) + y*s;

        at(1,0) = xy*(1 - c) + z*s;
        at(1,1) = c + y2*(1 - c);
        at(1,2) = yz*(1 - c) - x*s;

        at(2,0) = xz*(1 - c) - y*s;
        at(2,1) = yz*(1 - c) + x*s;
        at(2,2) = c + z2*(1 - c);
    }

    return *this;
}

// -- matrix composition

float4x4& float4x4::translate(float x, float y, float z) {
    float4x4 t(float4x4().translation(x, y, z));
    v512 = t.dot(*this).v512;
    return *this;
}

float4x4& float4x4::rotate(float x, float y, float z, float c, float s) {
    float4x4 r(float4x4().rotation(x, y, z, c, s));
    v512 = r.dot(*this).v512;
    return *this;
}


// -- matrix operations

float4x4 float4x4::transpose() const {
    float4x4 r;
    size_t i, j;
    
    for(i=0; i<4; ++i)
    for(j=0;j<4; ++j)
        r.at(j,i) = at(i,j);
    
    return r;
}

float4x4 float4x4::dot(const float4x4& m) const {
    
    float4x4 r;
    size_t i, j, k;
    
    // 665.436 ns
    //float s;
    //for(i=0; i<4; ++i)
    //for(j=0; j<4; ++j) {
    //    s = 0;
    //    for(k=0; k<4; ++k)
    //        s += at(i,k)*m.at(k,j);
    //    r.at(i,j) = s;
    //}

    // 326.1 ns
    //float4x4 t(m, transposed());
    //for(i=0; i<4; ++i)
    //for(j=0; j<4; j++) {
    //    r.at(i,j) = dotps(t128[i], t.t128[j]);;
    //}
    
    // 287.8 ns
    float4x4 t(m, transposed());
    for(i=0; i<4; ++i)
    {
        r.at(i,0) = dotps(t128[i], t.t128[0]);
        r.at(i,1) = dotps(t128[i], t.t128[1]);
        r.at(i,2) = dotps(t128[i], t.t128[2]);
        r.at(i,3) = dotps(t128[i], t.t128[3]);
    }
    
    return r;
}

float4 float4x4::apply(const float4& f) const {
    __m128 r128;

    // 45.5 ns
    //r128[0] = at(0,0)*f.x + at(0,1)*f.y + at(0,2)*f.z + at(0,3);
    //r128[1] = at(1,0)*f.x + at(1,1)*f.y + at(1,2)*f.z + at(1,3);
    //r128[2] = at(2,0)*f.x + at(2,1)*f.y + at(2,2)*f.z + at(2,3);
    //r128[3] = 0;

    // 32.5 ns
    r128[0] = dotps(t128[0], f.m128) + at(0,3);
    r128[1] = dotps(t128[1], f.m128) + at(1,3);
    r128[2] = dotps(t128[2], f.m128) + at(2,3);
    r128[3] = 0;

    return r128;
}
     
        

float4 float4x4::rotate(const float4& f) const {
    __m128 r128;
    
//    r128[0] = at(0,0)*f.x + at(0,1)*f.y + at(0,2)*f.z;
//    r128[1] = at(1,0)*f.x + at(1,1)*f.y + at(1,2)*f.z;
//    r128[2] = at(2,0)*f.x + at(2,1)*f.y + at(2,2)*f.z;
//    r128[3] = 0;

    r128[0] = dotps(t128[0], f.m128);
    r128[1] = dotps(t128[1], f.m128);
    r128[2] = dotps(t128[2], f.m128);
    r128[3] = 0;
    
    return r128;
}

float4v float4x4::apply(const float4v& v) const {
    float4v rv(v);
    size_t i, n = v.size();
    
    for(size_t i=0; i<n; ++i)
        rv[i] = apply(v[i]);
    
    return rv;
}
   
// -- str

std::string float4x4::str() const {
    char buf[512];
    sprintf(
        buf, 
        "[[%f, %f, %f, %f] \n"
        " [%f, %f, %f, %f] \n"
        " [%f, %f, %f, %f] \n"
        " [%f, %f, %f, %f]]\n", 
        at(0,0), at(0,1), at(0,2), at(0,3),
        at(1,0), at(1,1), at(1,2), at(1,3),
        at(2,0), at(2,1), at(2,2), at(2,3),
        at(3,0), at(3,1), at(3,2), at(3,3)
    );
    return buf;
}
