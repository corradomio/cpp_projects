//
// Created by Corrado Mio on 24/03/2024.
//
#include <iostream>
#include <cstdlib>
#include <memory>
#include <cxxabi.h>
#include <cassert>
#include "stdx/tryexc.h"


using namespace stdx;


// --------------------------------------------------------------------------
// block_t
// --------------------------------------------------------------------------

static block_t* BLOCK_END = nullptr;
// reinterpret_cast<block_t*>(0xFFFFFFFFFFFFFFFFul);

thread_local block_t* block_t::_top = BLOCK_END;


// --------------------------------------------------------------------------
// demangle
// --------------------------------------------------------------------------

std::string stdx::demangle(char const* name) {

    int status = -4; // some arbitrary value to eliminate the compiler warning

    // enable c++11 by passing the flag -std=c++11 to g++
    std::unique_ptr<char, void(*)(void*)> res {
        abi::__cxa_demangle(name, NULL, NULL, &status),
        std::free
    };

    return (status == 0) ? static_cast<const char*>(res.get()) : name ;
}


#ifdef _C98

struct handle {
        char* p;
        handle(char* ptr) : p(ptr) { }
        ~handle() { std::free(p); }
    };

    std::string demangle(const char* name) {

        int status = -4; // some arbitrary value to eliminate the compiler warning

        handle result( abi::__cxa_demangle(name, NULL, NULL, &status) );

        return (status==0) ? result.p : name ;
    }

#endif


// --------------------------------------------------------------------------
// stdx::exception
// --------------------------------------------------------------------------

stdx::exception::exception() {
    _what = "<thrown an exception>";
    _fillstack();
}


stdx::exception::exception(const char* msg) {
    assert(msg);
    _what = msg;
    _fillstack();
}


stdx::exception::exception(const std::string& msg) {
    _what = msg;
    _fillstack();
}


stdx::exception::~exception() throw () {
    _callstack.clear();
}


// --------------------------------------------------------------------------

void stdx::exception::_fillstack() {
    for(block_t* top = block_t::_top; top != BLOCK_END; top=top->_prev)
        _callstack.push_back(*top);
}


stdx::exception& stdx::exception::setcs(const char *fi, int l, const char *fu) {
    block_t here;
    here._file = fi;
    here._line = l;
    here._function = fu;

    _callstack.insert(_callstack.begin(), here);
    return (*this);
}

// --------------------------------------------------------------------------

void stdx::exception::printstack() {
    size_t n,i;
    block_t* ti_ptr;

    std::cerr << "Exception "
              << typestr((*this))
              << " : "
              << what()
              << std::endl;

    n = _callstack.size();
    for(i=0; i<n; ++i) {
        ti_ptr = &_callstack[i];
        std::cerr << "    "
                  << ti_ptr->_file
                  << " : "
                  << ti_ptr->_line
                  << " @ "
                  << ti_ptr->_function
                  << std::endl;
    }
}

