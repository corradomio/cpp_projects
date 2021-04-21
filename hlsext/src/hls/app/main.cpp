#include <iostream>
#include <vector>
#include <string>
#include "stdx/exception.h"

extern void appmain(const std::vector<std::string>& apps);

int main(int argc, char** argv, char**env) {
    try {
        std::vector<std::string> args(argc);

        for(int i=0; i<argc; ++i)
            args.emplace_back(argv[i]);

        appmain(args);
    }
    catch(stdx::exception_t& e) {
        e.printstack();
        return 1;
    }
    catch(std::exception& e) {
        std::cerr << "Caught exception: " << e.what() << std::endl;
        return 1;
    }
    catch (int error_code) {
        std::cerr << "Caught exception code: " << error_code << std::endl;
        return error_code;
    }
    catch(...) {
        std::cerr << "Caught a unknown exception " << std::endl;
        return 1;
    }

    return 0;
}