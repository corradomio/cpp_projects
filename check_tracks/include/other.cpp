//
// Created by Corrado Mio on 19/09/2020.
//


//void crete_grids() {
//
//    create_grid(5, 0, grid_fname(5, 0));
//
//    std::vector<int> sides{5,10,20,50,100};
//    std::vector<int> intervals{1,5,10,15,30,60};
//
//    for(int side : sides) {
//        tbb::parallel_for_each(intervals.begin(), intervals.end(), [&](int interval) {
//            std::string fname = grid_fname(side, interval);
//            std::cout << fname << std::endl;
//
//            create_grid(side, interval, grid_fname(side, interval));
//        });
//    }
//
//}

//void load_grids() {
//
//    {
//        DiscreteWorld dworld;
//        load_grid(dworld, grid_fname(5, 0));
//    }
//
//    std::vector<int> sides{5,10,20,50,100};
//    std::vector<int> intervals{1,5,10,15,30,60};
//
//    for(int side : sides) {
//        tbb::parallel_for_each(intervals.begin(), intervals.end(), [&](int interval) {
//            DiscreteWorld dworld;
//            load_grid(dworld, grid_fname(side, interval));
//        });
//    }
//
//}

//void create_load_grid() {
//    std::string fname = grid_fname(100, 60);
//
//    create_grid(100, 60, fname);
//
//    DiscreteWorld dworld;
//    load_grid(dworld, fname);
//}

//void save_encounters(const DiscreteWorld& dworld, const std::string& id, int side, int interval) {
//    std::cout << "-- " << id << ", " << side << ", " << interval << " --" << std::endl;
//
//    std::string dir = stdx::format(
//        "D:/Projects.github/cpp_projects/check_tracks/encounters/%d_%d", side, interval);
//    boost::filesystem::create_directory(dir);
//
//    std::string fname = stdx::format("%s/%s_%d_%d.csv",
//                                      dir.c_str(),
//                                      id.c_str(),
//                                      side, interval);
//
//    std::ofstream ofs(fname);
//    ofs << "\"id\",\"latitude\",\"longitude\",\"date\",\"time\",\"encounters\"" << std::endl;
//
//    std::map<int, encounters_set_t> encs = dworld.get_encounters(id);
//    std::vector<int> tlist = stdx::keys(encs, true);
//    for(int t : tlist) {
//        encounters_set_t& eset = encs[t];
//        dwpoint_t pt = dworld.to_point(eset.get_coords());
//        if (eset.eids.size() > 1)
//            ofs << id <<"," << pt.str() << ",\"" << stdx::str(encs[t].eids, "|") << "\"" << std::endl;
//    }
//}

//void save_encounters(int side, int interval) {
//    DiscreteWorld dworld;
//    dworld.load(grid_fname(side, interval));
//
//    const std::vector<std::string>& ids = dworld.ids();
//    for(const std::string& id : ids) {
//        save_encounters(dworld, id, side, interval);
//    }
//}

//void save_encounters() {
//    std::vector<int> sides{5,10,20,50,100};
//    std::vector<int> intervals{1,5,10,15,30,60};
//
//    //save_encounters(100, 60);
//
//    save_encounters(5, 0);
//    for(int side : sides) {
//        tbb::parallel_for_each(intervals.begin(), intervals.end(), [&](int interval) {
//            save_encounters(side, interval);
//        });
//    }
//}

