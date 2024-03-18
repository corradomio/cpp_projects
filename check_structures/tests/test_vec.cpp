//
// Created by Corrado Mio on 24/02/2024.
//
#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>
#include "stdx//linalg/vector_op.h"
#include "stdx//linalg/matrix_op.h"

using namespace stdx::linalg;

TEST_CASE( "constructor", "[vector]" ) {
    REQUIRE( vector_t<float>().size() == 0 );
    REQUIRE( vector_t<float>(10).size() == 10 );

    vector_t<float> v(5);
    v = 3;
    REQUIRE( v.size() == 5 );
    REQUIRE( v[0] == 3 );
    REQUIRE( v[4] == 3 );
}

TEST_CASE( "assignment", "[vector]" ) {

    vector_t<float> v(5);
    v = 3;
    vector_t<float> u = v;

    REQUIRE( u.size() == 5 );
    REQUIRE( u[0] == 3 );
    REQUIRE( u[4] == 3 );

    u[1] = 1;
    REQUIRE( v[1] == 1 );
    REQUIRE( u[1] == 1 );

    REQUIRE( u._info->refc == 2 );
}