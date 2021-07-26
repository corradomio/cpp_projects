//
// Created by Corrado Mio (Local) on 23/06/2021.
//
#include <iostream>
#include <stdx/cxxopts.h>

using namespace cxxopts;

int main2(int argc, char** argv) {
    cxxopts::Options options("MyProgram", "One line description of MyProgram");
    options.add_options()
        ("d,debug", "Enable debugging") // a bool parameter
        ("i,integer", "Int param", cxxopts::value<int>()->default_value("2"))
        ("f,file", "File name", cxxopts::value<std::string>())
        ("v,verbose", "Verbose output", cxxopts::value<bool>()->default_value("false"))
        ;

    auto result = options.parse(argc, argv);

    for (auto r : result.unmatched())
        std::cout << r << std::endl;

    std::cout << result["i"].as<int>() << std::endl;

    std::cout << "Hello Cruel World" << std::endl;
    return 0;
}

