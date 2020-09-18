//
// Created by Corrado Mio on 17/09/2020.
//
#include <string>
#include <iostream>
#include <csvstream.h>

#include "../include/dworld.h"

using namespace boost::filesystem;
using namespace boost::gregorian;
using namespace boost::posix_time;
using namespace boost;
using namespace hls::khalifa::summer;


const std::string DATASET = R"(D:\Dropbox\2_Khalifa\Progetto Summer\Dataset_3months)";


void create_grid(int side, int interval, std::string fname) {

    DiscreteWorld dworld(side, interval);

    std::cout << "create_grid(" << side << "," << interval << ")" << std::endl;

    try {
        int count = 0;

        path p(DATASET);

        // 0,  1          2           3    4          5                  6      7      8           9
        // "","latitude","longitude","V3","altitude","date.Long.format","date","time","person.id","track.id"

        for (directory_entry &de : directory_iterator(p)) {
            if (de.path().extension() != ".csv")
                continue;

            //std::cout << "  " << de.path().string()<< std::endl;

            csvstream csvin(de.path().string());

            std::vector<std::string> row;

            while  (csvin >> row) {
                count += 1;

                //if (count%100000 == 0)
                //    std::cout << "    " << count << std::endl;

                std::string id = row[8];
                double latitude  = lexical_cast<double>(row[1]);
                double longitude = lexical_cast<double>(row[2]);

                date date = from_string(row[6]);
                time_duration duration = duration_from_string(row[7]);
                ptime timestamp(date, duration);

                dworld.add(id, latitude, longitude, timestamp);
            }
        }
        std::cout << "    " << count << std::endl;

        std::cout << "save in(" << fname << ")" << std::endl;
        dworld.save(fname);
        dworld.dump();
        std::cout << std::endl;
    }
    catch(std::exception& e) {
        std::cout << e.what() << std::endl;
    }

}


void load_grid(DiscreteWorld& dworld, const std::string& fname) {
    std::cout << "load from(" << fname << ")" << std::endl;
    dworld.load(fname);
    dworld.dump();
}