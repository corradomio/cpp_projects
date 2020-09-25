//
// Created by Corrado Mio on 23/09/2020.
//

#ifndef CHECK_TRACKS_OTHER_H
#define CHECK_TRACKS_OTHER_H

extern std::string grid_fname(int side, int interval);
extern std::vector<std::tuple<int, int>> make_params(bool skip50=false);
extern void crete_grids();
extern void load_grids();

#endif //CHECK_TRACKS_OTHER_H
