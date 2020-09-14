/*
 * File:   float4x4test.cpp
 * Author: Corrado Mio
 *
 * Created on May 15, 2015, 7:45:05 AM
 */

#include "../float4.hpp"
#include "float4x4test.h"

namespace CppUnit {
    template<>
    struct assertion_traits<hls::float4>
    {
        static bool equal( const hls::float4& x, const hls::float4& y )
        {
            return x.equal(y);
        }

        static std::string toString( const hls::float4& x )
        {
            return x.str();
        }
    };

    template<>
    struct assertion_traits<hls::float4x4>
    {
        static bool equal( const hls::float4x4& x, const hls::float4x4& y )
        {
            return x.equal(y);
        }

        static std::string toString( const hls::float4x4& x )
        {
            return x.str();
        }
    };
}

using namespace hls;

CPPUNIT_TEST_SUITE_REGISTRATION(float4x4test);

float4x4test::float4x4test() {
}

float4x4test::~float4x4test() {
}

void float4x4test::setUp() {
}

void float4x4test::tearDown() {
}

void float4x4test::testConstructor() {
    float4 t(1,-1,1);
    float4 o;
    float4 x(1,0,0);
    float4 y(1,0,0);
    float4 z(1,0,0);
    float4 r;
    float4x4  i = float4x4();
    float4x4 mz = float4x4().zero();
    float4x4 tr = float4x4().translation(-1,1,-1);
    float4x4 r0 = float4x4().rotation(1,0,0,  1,0);

    CPPUNIT_ASSERT_EQUAL(i, r0);

    r = t.apply(tr);
    CPPUNIT_ASSERT_EQUAL(r, o);
    
    r = t.apply(i);
    CPPUNIT_ASSERT_EQUAL(r, t);
    
    r = t.apply(mz);
    CPPUNIT_ASSERT_EQUAL(r, o);
}

void float4x4test::testRotations() {
    float4 xp(+1,0,0);
    float4 yp(0,+1,0);
    float4 zp(0,0,+1);
    float4 xn(-1,0,0);
    float4 yn(0,-1,0);
    float4 zn(0,0,-1);

    float4 r;
    float4x4 rx = float4x4().rotation(float4(1,0,0), 90, true);
    float4x4 ry = float4x4().rotation(float4(0,1,0), 90, true);
    float4x4 rz = float4x4().rotation(float4(0,0,1), 90, true);
    
    r = yp.apply(rx);
    CPPUNIT_ASSERT_EQUAL(zp, r);
    
    r = zp.apply(ry);
    CPPUNIT_ASSERT_EQUAL(xp, r);
    
    r = xp.apply(rz);
    CPPUNIT_ASSERT_EQUAL(yp, r);
    
    r = xp.apply(rx);
    CPPUNIT_ASSERT_EQUAL(xp, r);
    r = xp.apply(ry);
    CPPUNIT_ASSERT_EQUAL(zn, r);

    r = yp.apply(ry);
    CPPUNIT_ASSERT_EQUAL(yp, r);
    r = yp.apply(rz);
    CPPUNIT_ASSERT_EQUAL(xn, r);

    r = zp.apply(rz);
    CPPUNIT_ASSERT_EQUAL(zp, r);
    r = zp.apply(rx);
    CPPUNIT_ASSERT_EQUAL(yn, r);
}

