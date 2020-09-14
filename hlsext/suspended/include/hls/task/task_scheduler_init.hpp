//
// Created by Corrado Mio on 28/02/2016.
//

#ifndef TBBTEST_TASK_SCHEDULER_INIT_HPP
#define TBBTEST_TASK_SCHEDULER_INIT_HPP

#include "task_defs.hpp"

namespace hls {
namespace task {

    class task_scheduler_init : internal::no_copy
    {
    public:
        static const int deffered = -1;
        static const int automatic = 0;

        task_scheduler_init(int nthreads=automatic);

        void initialize(int num_threads);
        void initialize(int num_threads, size_t stack_size);
    };

}};



#endif //TBBTEST_TASK_SCHEDULER_INIT_HPP
