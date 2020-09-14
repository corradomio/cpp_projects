//
// Created by Corrado Mio on 08/03/2016.
//

#include <exception>
#include <malloc.h>
#include "hls/thread/thread.hpp"

using namespace hls::thread;

thread_local thread* current_thread;

void void_call() { }


thread::thread() {
    _thread = nullptr;
    pthread_attr_init(&_attr);
    _runnable = this;
    _running  = false;
}

thread::thread(const runnable* torun) {
    _thread = nullptr;
    pthread_attr_init(&_attr);
    _runnable = const_cast<runnable*>(torun);
    _running  = false;
}

thread& thread::set_stack_size(size_t stack_size) {
    if (stack_size == 0) return *this;
    pthread_attr_setstacksize(&_attr, stack_size);
    return *this;
}

thread& thread::set_runnable(const runnable *torun) {
    _runnable = const_cast<runnable*>(torun);
    return *this;
}

thread::~thread() {
    pthread_attr_destroy(&_attr);
}

void thread::run() { }


thread* this_thread() {
    return current_thread;
}


void* thread::execute(void *context) {
    current_thread = static_cast<thread*>(context);
    current_thread->this_run();
    return current_thread;
}


void thread::this_run() {
    _running = true;
    try {
        run();
    }
    catch(std::exception &e) { }
    catch(std::exception *e) { }
    catch(...) { }
    _running = false;
}