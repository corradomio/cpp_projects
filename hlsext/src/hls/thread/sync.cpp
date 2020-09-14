//
// Created by Corrado Mio on 16/03/2016.
//

#include "hls/thread/barrier.hpp"
#include "hls/thread/semaphore.hpp"
#include "hls/thread/condition.hpp"

using namespace hls::thread;

// --------------------------------------------------------------------------
// barrier
// --------------------------------------------------------------------------

void barrier::wait() {
    _mutex.lock();
    __sync_add_and_fetch(&_waiting, 1);

    while(!_interrupted && _waiting < _count)
        _event.wait(_mutex);

    _event.signal_all();
    _mutex.unlock();
}


// --------------------------------------------------------------------------
// semaphore
// --------------------------------------------------------------------------

bool semaphore::acquire() {
    _mutex.lock();
    while (!_interrupted && _acquired >= _count)
        _event.wait(_mutex);

    if (!_interrupted && _acquired < _count)
        __sync_add_and_fetch(&_acquired, 1);

    _mutex.unlock();
    return true;
}

void semaphore::release() {
    _mutex.lock();
    __sync_sub_and_fetch(&_acquired, 1);
    _event.signal_all();
    _mutex.unlock();
}


// --------------------------------------------------------------------------
// condition
// --------------------------------------------------------------------------

void condition::set() {
    _mutex.lock();
    _event.signal_all();
    _mutex.unlock();
}

void condition::wait() {
    _mutex.lock();
    while(!_interrupted && !_cond)
        _event.wait(_mutex);
    _mutex.unlock();
}


// --------------------------------------------------------------------------
// end
// --------------------------------------------------------------------------
