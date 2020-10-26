//
// Created by Corrado Mio on 10/10/2020.
//

#include <iostream>
#include <stdx/properties.h>


int main() {
    stdx::properties props("D:\\Projects.github\\cpp_projects\\check_hlsext\\args.properties");

    for(const std::string& k : props.names())
        std::cout << k << ":" << props.get(k) << std::endl;

    props.get_ints("pm");
}