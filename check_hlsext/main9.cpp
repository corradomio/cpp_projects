//
// Created by Corrado Mio on 10/10/2020.
//

#include <iostream>
#include <hls/util/properties.h>


int main() {
    hls::util::properties props = hls::util::properties::read("D:\\Projects.github\\cpp_projects\\check_hlsext\\args.properties");

    for(const std::string k : props.names())
        std::cout << k << ":" << props.get(k) << std::endl;

}