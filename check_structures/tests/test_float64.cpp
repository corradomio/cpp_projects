//
// Created by Corrado Mio on 08/03/2024.
//
#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>
#include "stdx//float64/vector_op.h"
#include "stdx//float64/matrix_op.h"
#include "stdx//float64/dot_op.h"

using namespace stdx::float64;

TEST_CASE( "constructor", "[vector]" ) {
    REQUIRE( vector_t().size() == 0 );
    REQUIRE( vector_t(10).size() == 10 );

    vector_t v(5);
    v = 3;
    REQUIRE( v.size() == 5 );
    REQUIRE( v[0] == 3 );
    REQUIRE( v[4] == 3 );
}

TEST_CASE( "assignment", "[vector]" ) {

    vector_t v(5);
    v = 3;
    vector_t u = v;

    REQUIRE( u.size() == 5 );
    REQUIRE( u[0] == 3 );
    REQUIRE( u[4] == 3 );

    u[1] = 1;
    REQUIRE( v[1] == 1 );
    REQUIRE( u[1] == 1 );

    REQUIRE( u.info->refc == 2 );
}

TEST_CASE( "comparison", "[vector]") {
    vector_t v1 = zeros(10);
    vector_t v2 = zeros(5);
    vector_t v3 = ones(10);

    REQUIRE( v1 != v2 );
    REQUIRE( v1 != v3 );
    REQUIRE( v2 != v3 );
}

TEST_CASE( "comparison", "[matrix]") {
    matrix_t m1 = zeros(10, 10);
    matrix_t m2 = zeros(5, 10);
    matrix_t m3 = ones(10, 10);
    matrix_t m4 = zeros(10, 5);

    REQUIRE( m1 != m2 );
    REQUIRE( m1 != m3 );
    REQUIRE( m1 != m4 );
    REQUIRE( m2 != m3 );
    REQUIRE( m2 != m4 );
    REQUIRE( m3 != m4 );
}

TEST_CASE( "dot1v", "[vector]" ) {

    vector_t u = range(10);
    vector_t v = ones(10);
    matrix_t m = identity(10);
    vector_t r;
    matrix_t a, b;

    REQUIRE(dot(u, v) == 55);

    r = dot(m, v);
    REQUIRE( r == v );

    r = dot(u, m);
    REQUIRE( r == u );

    b = range(10, 10);
    a = dot(m, b);
    REQUIRE(a == b);
}

TEST_CASE( "dot2v", "[vector]" ) {

    vector_t u = range(5);
    vector_t v = ones(10);
    matrix_t m = range(5, 10);
    vector_t r;

    r = dot(m, v);
    REQUIRE(r.size() == 5);

    r = dot(u, m);
    REQUIRE(r.size() == 10);
}

TEST_CASE( "cross", "[vector]" ) {

    vector_t u = range(5);
    vector_t v = ones(10);
    matrix_t m = cross(u, v);

    REQUIRE(m.rows() == u.size());
    REQUIRE(m.cols() == v.size());
}


TEST_CASE( "dot1m", "[matrix]" ) {
    // A.B
    matrix_t A = range(5, 3);
    matrix_t B = range(3, 2);

    matrix_t C = dot(A, B);

    // {
    //  {22, 28},
    //  {49, 64},
    //  {76, 100},
    //  {103, 136},
    //  {130, 172}
    // }

    REQUIRE(C.rows() == A.rows());
    REQUIRE(C.cols() == B.cols());
    REQUIRE(C[0,0] == 22);
    REQUIRE(C[4,1] == 172);
}


TEST_CASE( "dot2m", "[matrix]" ) {
    // A^T.B
    matrix_t A = range(3, 5);
    matrix_t B = range(3, 2);

    matrix_t C = tdot(A, B);

    // {
    //  {74, 92},
    //  {83, 104},
    //  {92, 116},
    //  {101, 128},
    //  {110, 140}
    // }

    REQUIRE(C.rows() == A.cols());
    REQUIRE(C.cols() == B.cols());
    REQUIRE(C[0,0] == 74);
    REQUIRE(C[4,1] == 140);
}


TEST_CASE( "dot3m", "[matrix]" ) {
    // A^T.B
    matrix_t A = range(5, 3);
    matrix_t B = range(2, 3);

    matrix_t C = dott(A, B);

    // {
    //  {14, 32},
    //  {32, 77},
    //  {50, 122},
    //  {68, 167},
    //  {86, 212}
    // }

    REQUIRE(C.rows() == A.rows());
    REQUIRE(C.cols() == B.rows());
    REQUIRE(C[0,0] == 14);
    REQUIRE(C[4,1] == 212);
}