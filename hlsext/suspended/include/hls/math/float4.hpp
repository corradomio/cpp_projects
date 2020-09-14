/* 
 * File:   float4.hpp
 * Author: 
 *
 * Created on May 12, 2015, 7:28 PM
 */

/*
 * XMM0 .. XMM15: 128bit registers
 * MXCRS: 
 * 
 * 
 * SSE
 * -----------------------------------------------------------------
 * SSE data type: float[4]
 * 
 * MOVAPS:  m->r, r->r, r->m aligned    float[4]
 * MOVUPS:  m->r, r->r, r->m unaligned  float[4]
 * MOVSS:   m->r, r->r, r->m            float[1]
 * MOVLPS:  m->r, r->m       low        float[2]
 * MOVHPS:  m->r, r->m       high       float[2]
 * MOVLHPS: r->r             low->high  float[2]
 * MOVHLPS: r->r             high->low  float[2]
 * MOVMSKPS:
 * 
 * ADDPS, SUBPS, MULPS, DIVPS, RCPPS, SQRTPS, RSQRTPS, MAXPS, MINPS, 
 * ADDSS, SUBSS, MULSS, DIVSS, RCPSS, SQRTSS, RSQRTSS, MAXSS, MINSS, 
 * 
 * SHUFPS, UNPCKHPS, UNPCKLPS
 * 
 * 
 * SSE2
 * -----------------------------------------------------------------
 * 
 * SSE3
 * -----------------------------------------------------------------
 * 
 * MOVSHDUP, MOVSLDUP
 * ADDSUBPS, HADDPS, HSUBPS
 * 
 * 
 * SSSE3
 * -----------------------------------------------------------------
 * 
 * 
 * SSE4.1
 * -----------------------------------------------------------------
 * 
 * DPPS
 * 
 * ROUNDPS
 * ROUNDSS
 * 
 * BLENDPS, BLENDVPS
 * 
 * INSERTPS, EXTRACTPS
 * 
 * 
 * SSE4.2
 * -----------------------------------------------------------------
 * 
 */

#ifndef FLOAT4_HPP
#define	FLOAT4_HPP

#include <cmath>
#include <string>
#include <vector>
#include <intrin.h>


namespace hls {
namespace math {
    
    //typedef float __m128 __attribute__ ((__vector_size__ (16)));
    //typedef uint32_t __v4si __attribute__ ((__vector_size__ (16)));

    class float4;
    class float4x4;
    class float4v;

    // ======================================================================
    // float4
    // ======================================================================
    
    struct float4 {
    private:
        float4(const __m128& m): m128(m) { v[3] = 0; }

        friend class float4x4;
    public:
        union {
            struct { float v[4]; };
            struct { float x,y,z,t; };
            struct { __m128 m128; };
        };
    public:
        // -- constructors
        
        float4() { }
        explicit float4(float s): x(s), y(s), z(s), t(0) { }
        float4(float x,float y,float z): x(x), y(y), z(z), t(0) { }
        float4(const float4& f): m128(f.m128) { }
        
        // -- accessors
        
        float  operator[](size_t i) const { return v[i]; }
        float& operator[](size_t i)       { return v[i]; }
        
        // -- assignments
        
        float4& operator =(const float4& f) {
            m128 = f.m128;
            //x = f.x;
            //y = f.y;
            //z = f.z;
            return *this;
        }
        
        float4& operator +=(const float4& f) {
            // 10.4 ns
            //x += f.x;
            //y += f.y;
            //z += f.z;
            
            // 4.8 ns
            m128 = _mm_add_ps(m128, f.m128);

            // 4.8 ns
            //__asm__ (
            //    "movaps %[a], %%xmm0;"
            //    "addps  %[b], %%xmm0;"
            //    "movaps %%xmm0, %[ret];"
            //    : [ret] "+m" (m128)
            //    : [a] "m" (m128), [b] "m" (f.m128)
            //);
            return *this;
        }
        
        float4& operator -=(const float4& f) {
            // 10.2 ns
            //x -= f.x;
            //y -= f.y;
            //z -= f.z;

            // 4.8 ns
            m128 = _mm_sub_ps(m128, f.m128);

            // 4.8 ns
            //__asm__ (
            //    "movaps %[a], %%xmm0;"
            //    "subps  %[b], %%xmm0;"
            //    "movaps %%xmm0, %[ret];"
            //    : [ret] "+m" (m128)
            //    : [a] "m" (m128), [b] "m" (f.m128)
            //);
            return *this;
        }
        
        float4& operator *=(float s) {
            // 8 ns
            //x *= s;
            //y *= s;
            //z *= s;

            // 8.2 ns
            m128 = _mm_mul_ps(m128, _mm_set1_ps(s));

            // 8.2 ns
            //__m128 s128 = (__m128){ s, s, s, 0 };
            //__asm__ (
            //    "movaps %[a], %%xmm0;"
            //    "mulps  %[b], %%xmm0;"
            //    "movaps %%xmm0, %[ret];"
            //    : [ret] "+m" (m128)
            //    : [a] "m" (m128), [b] "m" (s128)
            //);
            return *this;
        }
        
        // -- operators
        
        float4 operator +() const { return float4( x, y, z); }
        float4 operator -() const { return float4(-x,-y,-z); }
        
        float4 operator +(const float4& f) const {
            // 12.6 ns
            //return float4(x + f.x, y + f.y, z + f.z);
            
            // 11.1 ns
            __m128 r128 = _mm_add_ps(m128, f.m128);
            
            // 10.0 ns
            //__m128 r128;
            //__asm__ (
            //    "movaps %[a], %%xmm0;"
            //    "addps  %[b], %%xmm0;"
            //    "movaps %%xmm0, %[ret];"
            //    : [ret] "=m" (r128)
            //    : [a] "m" (m128), [b] "m" (f.m128)
            //);
            return r128;
        }
        
        float4 operator -(const float4& f) const {
            // 12.8 ns
            //return float4(x - f.x, y - f.y, z - f.z);
            
            // 11.6 ns
            __m128 r128 = _mm_sub_ps(m128, f.m128);
            
            // 10.1 ns
            //__m128 r128;
            //__asm__ (
            //    "movaps %[a], %%xmm0;"
            //    "subps  %[b], %%xmm0;"
            //    "movaps %%xmm0, %[ret];"
            //    : [ret] "=m" (r128)
            //    : [a]  "m" (m128), [b] "m" (f.m128)
            //);
            return r128;
        }
        
        float4 operator *(float s) const {
            // 11.8 ns
            //return float4(x*s, y*s, z*s);
            
            // 12.5 ns
            __m128 r128 = _mm_sub_ps(m128, _mm_set1_ps(s));
            
            // 11.5 ns
            //__m128 r128;
            //__m128 s128 = (__m128){ s, s, s, 0 };
            //__asm__ (
            //    "movaps %[a], %%xmm0;"
            //    "mulps  %[b], %%xmm0;"
            //    "movaps %%xmm0, %[ret];"
            //    : [ret] "+m" (r128)
            //    : [a] "m" (m128), [b] "m" (s128)
            //);            
            return r128;
        }
        
        // -- predicates
        
        bool zero() const;
        
        bool equal(const float4& f) const;
        
        bool operator ==(const float4& f) const { return equal(f); }
        
        // -- scalar functions
        
        float dot(const float4& f) const;
        
        float norm() const;
        
        // -- vectorial functions
        
        float4 cross(const float4& f) const;
        
        // -- ortho
        
        float4 ortho() const;
        float4 ortho(const float4& v) const;
        float4 ortho(const float4& u, const float4& v) const;
        
        // -- utilities
        
        std::string str() const;
        
    };
    
    inline float4 operator *(float s, const float4& f) {
        return f*s;
    }
    
    inline float abs(const float4& f) {
        return f.norm(); 
    }
    
    inline float4 unit(const float4& f) {
        return f*(1./f.norm());
    }
    
    // ======================================================================
    // float4v
    // ======================================================================
    
    class float4v {
        std::vector<float4> data;
    public:
        float4v() { }
        
        //float4v(float4v&& v): data(v.data) { }
        float4v(const float4v& v): data(v.data) { }
        
        float4  at(size_t i) const { return data.at(i); }
        float4& at(size_t i)       { return data.at(i); }
        
        float4  operator[](size_t i) const { return data[i]; }
        float4& operator[](size_t i)       { return data[i]; }
        
        float4v& add(const float4& f) {
            data.push_back(f);
            return *this;
        }
        
        size_t size() const { return data.size(); }
    };
    
    // ======================================================================
    // f4
    // ======================================================================
    
    struct f4 {
        static float4 origin;
        static float4 x_axis;
        static float4 y_axis;
        static float4 z_axis;
        static float4 neg_x_axis;
        static float4 neg_y_axis;
        static float4 neg_z_axis;
    };
    
    // ======================================================================
    // float4x4
    // ======================================================================
    
    typedef float __v512 __attribute__ ((__vector_size__ (64), __may_alias__));
    
    class float4x4 {
        union {
            struct { float m[4][4]; };
            struct { __m128 t128[4]; };
            struct { __v512 v512; };
        };
        
        struct transposed { };
        
        float4x4(const float4x4& t, transposed);
    public:
        float4x4();
        float4x4(const float4x4& t): v512(t.v512) { }
        
        float4x4(const float* t);
        
        // -- constructors
        
        float4x4& zero();
        float4x4& identity();
        float4x4& translation(float x, float y, float z);
        float4x4& rotation(float x, float y, float z, float c, float s);
        
        
        float4x4& translation(const float4& t) {
            return translation(t.x, t.y, t.z);
        }
        
        float4x4& rotation(float x, float y, float z, float a, bool deg) {
            a = deg ? a*0.017453292519943295f : a;
            return rotation(x, y, z, cosf(a), sinf(a));
        }

        float4x4& rotation(const float4& v, float a, bool deg) {
            a = deg ? a*0.017453292519943295f : a;
            return rotation(v.x, v.y, v.z, cosf(a), sinf(a));
        }

        float4x4& rotation(const float4& v, float c, float s) {
            return rotation(v.x, v.y, v.z, c, s);
        }

        // -- accesors
        
        inline float  at(size_t i, size_t j) const { return m[i][j]; }
        inline float& at(size_t i, size_t j)       { return m[i][j]; }
        
        // -- matrix composition
        
        float4x4& translate(float x, float y, float z);
        
        float4x4& rotate(float x, float y, float z, float c, float s);
        
        
        float4x4& translate(const float4& t) {
            return translate(t.x, t.y, t.z);
        }

        float4x4& rotate(float x, float y, float z, float a, bool deg) {
            a = deg ? a*0.017453292519943295f : a;
            return rotate(x, y, z, cosf(a), sinf(a));
        }

        float4x4& rotate(const float4& v, float a, bool deg) {
            a = deg ? a*0.017453292519943295f : a;
            return rotate(v.x, v.y, v.z, cosf(a), sinf(a));
        }

        float4x4& rotate(const float4& v, float c, float s) {
            return rotate(v.x, v.y, v.z, c, s);
        }
        
        // -- matrix operations
        
        float4x4 transpose() const;
        float4x4 dot(const float4x4& t) const;
        
        // -- vector operations
        
        float4  apply( const float4&  f) const;
        float4  rotate(const float4&  f) const;
        
        float4v apply( const float4v& f) const;
        
        // -- utilities
        
        std::string str() const;
        
    };

    // ======================================================================
    // end
    // ======================================================================

}}

#endif	/* FLOAT4_HPP */

