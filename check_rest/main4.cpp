//
// Created by Corrado Mio on 03/10/2020.
//
#include <iostream>
#include <array>
#include <cereal/archives/json.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/array.hpp>

int main4()
{
    cereal::JSONOutputArchive archive( std::cout );
    //bool arr[] = {true, false};

    std::array<bool,2> arr{true, false};
    std::vector<int>   vec{1, 2, 3, 4, 5};
    archive( CEREAL_NVP(vec), CEREAL_NVP(arr) );
}
