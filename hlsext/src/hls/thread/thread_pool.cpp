//
// Created by Corrado Mio on 07/03/2016.
//

#include <iostream>
#include "hls/thread/thread_pool.hpp"


using namespace hls::thread;

// --------------------------------------------------------------------------
// worker_thread
// --------------------------------------------------------------------------

worker_thread& worker_thread::set_pool(thread_pool* pool) {
    _pool = pool;
    return *this;
}


void worker_thread::run() {

    while(!interrupted()) {
        runnable* torun = _pool->dequeue();
        if (torun == nullptr)
            continue;

        try {
            _pool->starting();
            torun->run();
            _pool->finished();
        }
        catch(...) {
            _pool->finished();
            throw;
        }
    }

}

// --------------------------------------------------------------------------
// hls::thread::thread_pool
// --------------------------------------------------------------------------

thread_pool::thread_pool()
: sync()
{
    _waiting = 0;
    _stack_size = 0;
    _queue_length = 0;
    _num_threads = hardware_concurrency();
}

thread_pool::~thread_pool() {
    stop();
}

void thread_pool::start() {
    _running = 0;
    _interrupted = false;

    _threads.resize(_num_threads);
    for(size_t i=0; i<_num_threads; ++i)
        _threads[i].set_pool(this)
                   .set_stack_size(_stack_size)
                   .start();
}

void thread_pool::stop() {
    _interrupted = true;
    for(size_t i=0; i<_num_threads; ++i)
        _threads[i].interrupt();

    _mutex.lock();
    _event.signal_all();
    _mutex.unlock();

    for(size_t i=0; i<_num_threads; ++i)
        _threads[i].join();

    _threads.clear();
    _num_threads = 0;
}


void thread_pool::join() {
    _mutex.lock();
    while(!_interrupted && (_running + _waiting) > 0)
        _event.wait(_mutex);
    _mutex.unlock();
}

void thread_pool::enqueue(runnable* runnable) {
    _mutex.lock();
    _works.push(runnable);

    __sync_add_and_fetch(&_waiting, 1);

    _event.signal();
    _mutex.unlock();
}

runnable* thread_pool::dequeue() {
    _mutex.lock();
    while(!_interrupted && _waiting == 0)
        _event.wait(_mutex);

    runnable* torun = nullptr;
    if (!_interrupted && _waiting > 0) {
        torun = _works.front();
        _works.pop();

        __sync_sub_and_fetch(&_waiting, 1);
    }
    _mutex.unlock();

    return torun;
}

void thread_pool::starting() {
    __sync_add_and_fetch(&_running, 1);
}

void thread_pool::finished() {
    __sync_sub_and_fetch(&_running, 1);

    _mutex.lock();
    _event.signal_all();
    _mutex.unlock();
}

// --------------------------------------------------------------------------
// end
// --------------------------------------------------------------------------
