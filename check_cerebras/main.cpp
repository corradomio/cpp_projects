#include <iostream>
#include "cerebras.h"

using namespace cerebras;

int main() {
    color red = get_color(7);
    data_task_id red_task_id = get_data_task_id(red);

    f32 result = 0.0;

    std::cout << "Hello, World!" << std::endl;
    return 0;
}
