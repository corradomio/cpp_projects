#include <windows.h>
#include <fstream>
#include <iostream>
#include <algorithm>

int main()
{
    std::cout << "Hello, World!" << std::endl;

    std::string inputFile = "E:\\Datasets\\mbtiles\\monaco-latest.osm.pbf";
    std::ifstream infile(inputFile, std::ifstream::in);
    infile.close();
    return 0;
}
