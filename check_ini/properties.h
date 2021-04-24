//
// Created by Corrado Mio (Local) on 23/04/2021.
//

#ifndef CHECK_INI_PROPERTIES_H
#define CHECK_INI_PROPERTIES_H

namespace stdx {

    class properties {

    };

    properties load_init(const std::string& file);
    properties load_toml(const std::string& file);
}

#endif //CHECK_INI_PROPERTIES_H
