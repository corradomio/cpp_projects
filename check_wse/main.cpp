#include <iostream>
#include <stdx/intfloat.h>
#include "include/cerebras.h"

using namespace cerebras;


// struct main_task_t: public task {
//     f32 result;
//
//     main_task_t() = default;
//
//     void run(f32 wavelet_data) {
//         result = wavelet_data;
//     }
// };

f32 result = 0.0;
f32 sum = 0.0;

void main_task(f32 wavelet_data) {
    result = wavelet_data;
}

void foo_task() {
    sum += result;
}

int main() {
    std::cout << "Hello, World!" << std::endl;

    const color red = get_color(7);
    const input_queue iq = get_input_queue(2);
    const data_task_id red_task_id = get_data_task_id(iq);
    const data_task_id red_task_v2 = get_data_task_id(red);
    const local_task_id foo_task_id = get_local_task_id(8);

    bind_data_task(main_task, red_task_id);
    bind_local_task(foo_task, foo_task_id);

    return 0;
}
