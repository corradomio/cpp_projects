
#include "other.h"


// --------------------------------------------------------------------------

// --------------------------------------------------------------------------

int main() {
    //create_grids();
    //load_grids();

    std::vector<std::tuple<int, int>> params;
    params = make_params(true);

    //save_encounters(params);
    //save_slot_encounters();
    //save_time_encounters();

    simulate(params);

    //create_grid_test();
    //simulate_test();

    return 0;
}
