#include <iostream>
#include <vector>
#include <string>
#include "../../../include/hls/lang/exception.hpp"

extern void appmain(const std::vector<std::string>& apps);

int main(int argc, char** argv, char**env) {
    try {
        std::vector<std::string> args;

        for(int i=0; i<argc; ++i)
            args.push_back(argv[i]);

        appmain(args);
    }
    catch(hls::lang::exception_t& e) {
//        std::cerr << "Catched exception: " << e.what() << std::endl;
        e.printstack();
        return 1;
    }
    catch(std::exception& e) {
        std::cerr << "Catched exception: " << e.what() << std::endl;
        return 1;
    }
    catch (int error_code) {
        std::cerr << "Catched exception code: " << error_code << std::endl;
        return error_code;
    }
    catch(...) {
        std::cerr << "Catched a unknown exception " << std::endl;
        return 1;
    }

    return 0;
}