//
// Created by Corrado Mio on 20/02/2022.
//
#include<iostream>
#include<iomanip>
#include<fstream>

using namespace std;

int main(){

    ofstream outfile;
    outfile.open("nums.txt");

    for (int i = 15; i < 211; i++) {
        outfile << i << " ";
    }

    outfile.close();

    return 0;

}
