//
// Created by Corrado Mio on 02/06/2024.
//

#ifndef CEREBRAS_H
#define CEREBRAS_H

#include <thread>
#include <stdx/intfloat.h>
#include <stdx/language.h>

#define N_COLORS    24
#define N_TASK_IDS  64


namespace cerebras {

    // typedef int8_t  color;          // 0..23
    // typedef int8_t  input_queue;    // ...
    // typedef int8_t  task_id;        // 0..63
    // typedef int8_t  data_task_id;   // 0..63

    struct color {
        int8_t id;

        color(): id(0){ }
        explicit color(int8_t id): id(id) { }
    };
    struct input_queue {
        int8_t id;
        explicit input_queue(int8_t id): id(id) { }
    };

    struct task_id {
        int8_t id;
        explicit task_id(int8_t id): id(id) { }
        void assign(const task_id& other) {
            self.id = other.id;
        }
    };
    struct data_task_id: public task_id {
        explicit data_task_id(int8_t id): task_id(id) { }
        data_task_id& operator =(const data_task_id& other) {
            self.assign(other);
            return self;
        }
    };
    struct local_task_id: public task_id {
        explicit local_task_id(int8_t id): task_id(id) { }
        local_task_id& operator =(const local_task_id& other) {
            self.assign(other);
            return self;
        }
    };

    struct wavelet_t {
        color c;
        int64_t data;

        wavelet_t() { }
    };

    enum direction_t {
        north, south, east, west,
        left=west,
        right=east,
        up=north,
        down=south
    };

    enum task_type {
        data_task,
        local_task,
        control_task
    };

    struct queue_t {
        wavelet_t  data;

        queue_t() { }

        void set(wavelet_t  data);
        wavelet_t get();
    };

    // 24 routable colors

    struct router {
        const int n_colors = N_COLORS;
        queue_t colors[N_COLORS];

        router() { }

        wavelet_t get(color c);
        void send(direction_t d, color c, wavelet_t w);
    };

    // processing element
    //      router
    //      compute engine/processor
    //
    //      class members: local data
    //      class methods: local code
    //      task
    //
    //  task_id:            0..63
    //  routable task:      0..23   WSE/2
    //                      0..7    WSE/3

    //
    struct pe_t {
        task_id     id;
        task_type   type;
        router      r;
        bool _activated;
        bool _blocked;

        pe_t(task_id& id): id(id) {
            _activated = false;
            _blocked = false;
        }

        void activated(bool a) { self._activated = a; }
        void blocked(bool b){ self._blocked = b; }

        virtual void run() { };
    };

    struct task: public pe_t {

    };


    class wafer_scale_engine_t {
        size_t rows;
        size_t cols;
    public:
        wafer_scale_engine_t(size_t rows, size_t col);
        void send(color color, wavelet_t data);
    };

    extern wafer_scale_engine_t WSE;

    inline color get_color(int c) {
        return color(c);
    }

    inline input_queue get_input_queue(int iq) {
        return input_queue(iq);
    }

    inline data_task_id get_data_task_id(input_queue iq) {
        return data_task_id(iq.id);
    }

    inline data_task_id get_data_task_id(color c) {
        return data_task_id(c.id);
    }
    inline local_task_id get_local_task_id(int id) {
        return local_task_id(id);
    }

    inline void bind_data_task( void (*)(f32), data_task_id id){}
    inline void bind_local_task(void (*)(),   local_task_id id){}

    inline void initialize_queue(input_queue& iq, color c){}
    inline void block()

}

#endif //CEREBRAS_H
