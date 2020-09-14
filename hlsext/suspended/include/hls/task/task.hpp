//
// Created by Corrado Mio on 28/02/2016.
//

#ifndef TBBTEST_TASK_HPP
#define TBBTEST_TASK_HPP

#include <deque>

namespace hls {
namespace task {

    struct runnable {
        virtual void run() = 0;
    };

    struct task : public runnable
    {
        task() { }

        virtual void run () { }
    };

}};

#endif //TBBTEST_TASK_HPP
