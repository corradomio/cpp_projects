#include <iostream>
#include <stdx/once.h>
#include <stdx/trycatch.h>

int main() {
    _try
        std::cout << "_try" << std::endl;
        _throw(stdx::exception("boh"))
    _finally
        std::cout << "_finally" << std::endl;
    _end_finally

    _try {
        std::cout << "_try" << std::endl;
        _throw(stdx::exception("boh"))
    }
    _catch(stdx::exception& e) {
        std::cout << e.what() << std::endl;
    }
    _end_catch

    _try {
        std::cout << "Cruel" << std::endl;
    }
    _end_try

    _try {
        std::cout << "World" << std::endl;
    }
    _end_try


    return 0;
}

int main2() {
    std::cout << "Hello, World!" << std::endl;

    _try {
        stdx::once<int> i;
        i = 1;
        int j = i + 1;
//        i = 2;
    }
    _catch(stdx::exception& e) {
//        std::cerr << e.what();
        e.printstack();
    }
    _end_catch

    _try {
        int j=2;
    }
    _end_try


    return 0;
}
