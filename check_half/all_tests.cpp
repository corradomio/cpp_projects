//
// Created by Corrado Mio on 28/07/2023.
//
#include "ieee754.h"

#define CATCH_CONFIG_MAIN // provides main(); this line is required in only one .cpp file
// #include "catch_amalgamated.hpp"
#include <catch2/catch_test_macros.hpp>

// int theAnswer() { return 6*9; } // function to be tested
//
// TEST_CASE( "Life, the universe and everything", "[42][theAnswer]" ) {
//     REQUIRE(theAnswer() == 42);
// }

using namespace ieee754;

TEST_CASE("float32", "[t+2]") {
    float32 f1{1};
    float32 f2{1};
    float32 fr = f1 + f2;

    REQUIRE(fr == 2);
}
TEST_CASE("float32", "[t+1.5]") {
    float32 f1{1};
    float32 f2{1.5};
    float32 f3{2.5};

    float32 fr = f1+f2;

    REQUIRE(fr == 2.5);
}
TEST_CASE("float32", "[t+1.9]") {
    float32 f1{0,127,float32::M_MAX-1};
    float32 f2{0,127,float32::M_MAX-1};
    float32 f3{0, 128, float32::M_MAX-1};

    float32 fr = f1+f2;

    REQUIRE(fr == f3);
}