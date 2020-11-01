//
// Created by Corrado Mio on 23/09/2020.
//

#include <tuple>
#include <string>
#include <vector>
#include <stdx/properties.h>

#ifndef CHECK_TRACKS_OTHER_H
#define CHECK_TRACKS_OTHER_H

extern std::string grid_fname(const std::string& worlds_dir, int side, int interval);

extern void encounters(const stdx::properties& props);
extern void world(const stdx::properties& props);
extern void infected_users(const stdx::properties& props);
extern void simulate(const stdx::properties& props);

extern void save_infected(const std::string file, const std::vector<std::unordered_set<int>>&  vinfected);
extern void load_infected(const std::string file,       std::vector<std::unordered_set<int>>&  vinfected);

#endif //CHECK_TRACKS_OTHER_H
