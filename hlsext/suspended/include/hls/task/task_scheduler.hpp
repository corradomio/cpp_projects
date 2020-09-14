//
// Created by Corrado Mio on 28/02/2016.
//

#ifndef TBBTEST_TASK_SCHEDULER_HPP
#define TBBTEST_TASK_SCHEDULER_HPP

namespace hls {
namespace task {

    struct task;

    class task_queue {
        size_t _max_queue;
        std::deque<task*> _task_queue;
    public:
        size_t no_limits = 0;
    public:
        task_queue(size_t max_queue = no_limits): _max_queue(max_queue) { }

        void enqueue(task* atask);

        task* dequeue();
    };


    class task_scheduler
    {
    public:
        task_scheduler(int num_threads, size_t stack_size);
    };

}};

#endif //TBBTEST_TASK_SCHEDULER_HPP
