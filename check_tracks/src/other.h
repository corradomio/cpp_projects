//
// Created by Corrado Mio on 23/09/2020.
//

#include <tuple>
#include <vector>

#ifndef CHECK_TRACKS_OTHER_H
#define CHECK_TRACKS_OTHER_H

extern std::string grid_fname(int side, int interval);
extern std::vector<std::tuple<int, int>> make_params(bool skip50=false);

extern void create_grids();
extern void load_grids();
extern void save_encounters(const std::vector<std::tuple<int, int>>& params);

extern void simulate(const std::vector<std::tuple<int, int>>& params);

extern void create_grid_test();
extern void simulate_test();

#endif //CHECK_TRACKS_OTHER_H
