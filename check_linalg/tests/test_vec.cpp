//
// Created by Corrado Mio on 24/02/2024.
//
#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>
#include "../linalg/linalg.h"

using namespace stdx::linalg;

TEST_CASE( "constructor", "[vector]" ) {
    REQUIRE( vector().size() == 0 );
    REQUIRE( vector(10).size() == 10 );

    vector v(3, 5);
    REQUIRE( v.size() == 5 );
    REQUIRE( v[0] == 3 );
    REQUIRE( v[4] == 3 );
}

TEST_CASE( "assignment", "[vector]" ) {
    vector v(3, 5);
    vector u = v;

    REQUIRE( u.size() == 5 );
    REQUIRE( u[0] == 3 );
    REQUIRE( u[4] == 3 );

    u[1] = 1;
    REQUIRE( v[1] == 1 );
    REQUIRE( u[1] == 1 );

    REQUIRE( u.p->refc == 2 );
}