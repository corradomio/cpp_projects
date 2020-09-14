/*
 * File:   float4testclass.h
 * Author: Corrado Mio
 *
 * Created on May 14, 2015, 9:21:19 PM
 */

#ifndef FLOAT4TESTCLASS_H
#define	FLOAT4TESTCLASS_H

#include <cppunit/extensions/HelperMacros.h>

class float4test : public CPPUNIT_NS::TestFixture {
    CPPUNIT_TEST_SUITE(float4test);

    CPPUNIT_TEST(testConstructor);
    CPPUNIT_TEST(testArithmetic);
    CPPUNIT_TEST(testProducts);
    CPPUNIT_TEST(testOrtho);

    CPPUNIT_TEST_SUITE_END();

public:
    float4test();
    virtual ~float4test();
    void setUp();
    void tearDown();

private:
    void testMethod();
    void testFailedMethod();
        
    void testConstructor();
    void testArithmetic();
    void testProducts();
    void testOrtho();

};

#endif	/* FLOAT4TESTCLASS_H */

