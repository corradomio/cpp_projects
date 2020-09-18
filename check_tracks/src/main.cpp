#include <string>
#include <iostream>
#include <boost/filesystem.hpp>
//#include <gc_cpp.h>

using namespace boost::filesystem;

const std::string DATASET = R"(D:\Dropbox\2_Khalifa\Progetto Summer\Dataset)";

struct coords {
    int i, j, t;
};

int main() {
    std::cout << "Hello, World!" << std::endl;

    path p(DATASET);

    for (directory_entry& de : directory_iterator(p)) {
        if (de.path().extension() == ".csv")
            std::cout << de.path() << std::endl;
    }

    return 0;
}
