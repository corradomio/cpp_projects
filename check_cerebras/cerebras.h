//
// Created by Corrado Mio on 17/03/2024.
//

#ifndef CHECK_CEREBRAS_CEREBRAS_H
#define CHECK_CEREBRAS_CEREBRAS_H

#include <stdint.h>

namespace cerebras {

    typedef int8_t   i8;
    typedef int16_t  i16;
    typedef int32_t  i32;
    typedef int64_t  i64;
    typedef __int128 i128;
    typedef uint8_t  u8;
    typedef uint16_t u16;
    typedef uint32_t u32;
    typedef uint64_t u64;
    typedef unsigned __int128 u128;
    typedef float f32;
    typedef double f64;
    typedef long double f128;

    struct task_id { uint8_t id; task_id(uint8_t id): id(id) { } };
    struct data_task_id    : public task_id {    data_task_id(uint8_t id): task_id(id) { } };
    struct local_task_id   : public task_id {   local_task_id(uint8_t id): task_id(id) { } };
    struct control_task_id : public task_id { control_task_id(uint8_t id): task_id(id) { } };

    struct task {
    protected:
        task_id id;
        bool routable;
        bool activable;
        task(task_id id): id(id) { }
    public:
        virtual void main() = 0;
        virtual void main(uint32_t wavelet_data) = 0;
    };

    struct data_task : public task {
        data_task(data_task_id id): task(id) { }
    };

    struct local_task : public task {
        local_task(local_task_id id): task(id) { }
    };

    struct control_task : public task {
        control_task(control_task_id id): task(id) { }
    };


    struct color {
        uint8_t c;
        color(uint8_t c): c(c){ }
    };

    struct wavelet {
        uint32_t data;
    };

    inline color get_color(uint8_t c) { return {c}; }
    inline data_task_id get_data_task_id(color c) {
        return {c.c};
    }
}

#endif //CHECK_CEREBRAS_CEREBRAS_H
