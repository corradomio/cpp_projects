#include <iostream>
#include <intrin.h>
#include <x86gprintrin.h>
using namespace std;


int main() {
    std::cout << "Hello, World!" << std::endl;
    unsigned short us[3] = {0, 0xFF, 0xFFFF};
    unsigned short usr;
    unsigned int   ui[4] = {0, 0xFF, 0xFFFF, 0xFFFFFFFF};
    unsigned int   uir;

    for (int i=0; i<3; i++) {
        usr = __lzcnt16(us[i]);
        cout << "__lzcnt16(0x" << hex << us[i] << ") = " << dec << usr << endl;
    }

    for (int i=0; i<4; i++) {
        uir = __lzcnt64(ui[i]);
        cout << "__lzcnt(0x" << hex << ui[i] << ") = " << dec << uir << endl;
    }

    return 0;
}
