/*
 * File:   float4testclass.cpp
 * Author: Corrado Mio
 *
 * Created on May 14, 2015, 9:21:19 PM
 */

#include "../float4.hpp"
#include "float4test.h"

using namespace hls;

namespace CppUnit {
    template<>
    struct assertion_traits<hls::float4>
    {
        static bool equal( const hls::float4& x, const hls::float4& y )
        {
            return x == y;
        }

        static std::string toString( const hls::float4& x )
        {
            return x.str();
        }
    };
}


CPPUNIT_TEST_SUITE_REGISTRATION(float4test);

float4test::float4test() {
}

float4test::~float4test() {
}

void float4test::setUp() {
}

void float4test::tearDown() {
}


void float4test::testConstructor() {
    float4 c(0);
    float4 a(1,2,3);
    float4 b(a);
    
    CPPUNIT_ASSERT_EQUAL(float4(0), c);

    CPPUNIT_ASSERT_EQUAL(1.f, a.x);
    CPPUNIT_ASSERT_EQUAL(2.f, a.y);
    CPPUNIT_ASSERT_EQUAL(3.f, a.z);

    CPPUNIT_ASSERT_EQUAL(1.f, a[0]);
    CPPUNIT_ASSERT_EQUAL(2.f, a[1]);
    CPPUNIT_ASSERT_EQUAL(3.f, a[2]);

    CPPUNIT_ASSERT_EQUAL(1.f, b[0]);
    CPPUNIT_ASSERT_EQUAL(2.f, b[1]);
    CPPUNIT_ASSERT_EQUAL(3.f, b[2]);
    
    c = b;

    CPPUNIT_ASSERT_EQUAL(1.f, c[0]);
    CPPUNIT_ASSERT_EQUAL(2.f, c[1]);
    CPPUNIT_ASSERT_EQUAL(3.f, c[2]);
}


void float4test::testArithmetic()
{
    float4 a(1,2,3);
    float4 b(3,2,1);
    float4 c(4,4,4);
    float4 d(2,4,6);
    float4 pa(+1,+2,+3);
    float4 na(-1,-2,-3);
    float4 t, z;
    
    t = a + b;
    CPPUNIT_ASSERT_EQUAL(c, t);

    t = c - b;
    CPPUNIT_ASSERT_EQUAL(a, t);

    t = z;
    t += a;
    CPPUNIT_ASSERT_EQUAL(a, t);
    
    t -= a;
    CPPUNIT_ASSERT_EQUAL(z, t);
    
    t = +a;
    CPPUNIT_ASSERT_EQUAL(pa, t);
    
    t = -a;
    CPPUNIT_ASSERT_EQUAL(na, t);
    
    t = a*2;
    CPPUNIT_ASSERT_EQUAL(d, t);

    t = 2*a;
    CPPUNIT_ASSERT_EQUAL(d, t);
    
    t *= 0.5;
    CPPUNIT_ASSERT_EQUAL(a, t);
}

void float4test::testProducts()
{
    float4 t(1,2,3);
    float4 xp = f4::x_axis;
    float4 yp = f4::y_axis;
    float4 zp = f4::z_axis;
    float4 xn = f4::neg_x_axis;
    float4 yn = f4::neg_y_axis;
    float4 zn = f4::neg_z_axis;
    
    CPPUNIT_ASSERT_EQUAL(14.f, t.dot(t));
    CPPUNIT_ASSERT_EQUAL(1.f, xp.dot(xp));

    CPPUNIT_ASSERT_EQUAL(zp, xp.cross(yp));
    CPPUNIT_ASSERT_EQUAL(xp, yp.cross(zp));
    CPPUNIT_ASSERT_EQUAL(yp, zp.cross(xp));

    CPPUNIT_ASSERT_EQUAL(zn, yp.cross(xp));
    CPPUNIT_ASSERT_EQUAL(xn, zp.cross(yp));
    CPPUNIT_ASSERT_EQUAL(yn, xp.cross(zp));
}

void float4test::testOrtho()
{
    float4 xp = f4::x_axis;
    float4 yp = f4::y_axis;
    float4 zp = f4::z_axis;
    float4 xn = f4::neg_x_axis;
    float4 yn = f4::neg_y_axis;
    float4 zn = f4::neg_z_axis;

    float4 v1(2,0,0);
    float4 v2(2,2,0);
    float4 v3(2,2,2);
    
    float4 b1 = v1.ortho();
    float4 b2 = v2.ortho(b1);
    float4 b3 = v3.ortho(b1,b2);

    CPPUNIT_ASSERT_EQUAL(xp, b1);
    CPPUNIT_ASSERT_EQUAL(yp, b2);
    CPPUNIT_ASSERT_EQUAL(zp, b3);
    
    
    float4 w1(-2,0,0);
    float4 w2(-2,-2,0);
    float4 w3(-2,-2,-2);
    
    float4 c1 = w1.ortho();
    float4 c2 = w2.ortho(b1);
    float4 c3 = w3.ortho(b1,b2);

    CPPUNIT_ASSERT_EQUAL(xn, c1);
    CPPUNIT_ASSERT_EQUAL(yn, c2);
    CPPUNIT_ASSERT_EQUAL(zn, c3);
}
