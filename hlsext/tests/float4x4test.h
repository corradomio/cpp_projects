/*
 * File:   float4x4test.h
 * Author: Corrado Mio
 *
 * Created on May 15, 2015, 7:45:05 AM
 */

#ifndef FLOAT4X4TEST_H
#define	FLOAT4X4TEST_H

#include <cppunit/extensions/HelperMacros.h>

class float4x4test : public CPPUNIT_NS::TestFixture {
    CPPUNIT_TEST_SUITE(float4x4test);

    CPPUNIT_TEST(testConstructor);
    CPPUNIT_TEST(testRotations);

    CPPUNIT_TEST_SUITE_END();

public:
    float4x4test();
    virtual ~float4x4test();
    void setUp();
    void tearDown();

private:
    void testConstructor();
    void testRotations();
};

#endif	/* FLOAT4X4TEST_H */

