#include <iostream>
#include <typeinfo>
#include <cstdlib>
#include <memory>
#include <cxxabi.h>
#include <cassert>
#include "../../../include/hls/lang/exception.hpp"


using namespace hls::lang;


// --------------------------------------------------------------------------
// block_t
// --------------------------------------------------------------------------

static block_t* BLOCK_END = reinterpret_cast<block_t*>(0xFFFFFFFFFFFFFFFFul);

thread_local block_t* block_t::_top = BLOCK_END;


block_t::block_t(const char* fi, int l, const char* fu)
        : _file(fi), _line(l), _function(fu),
          _prev(_top)
{
    _top = this;
}


block_t::~block_t()
{
    if (_prev) _top = _prev;
}


/// costructtori usati nella creazione del callstack
block_t::block_t()
        : _file(nullptr), _line(0), _function(nullptr),
          _prev(nullptr)
{

}

block_t::block_t(const block_t& b)
        : _file(b._file), _line(b._line), _function(b._function),
          _prev(nullptr)
{

}


block_t& block_t::operator =(const block_t& b)
{
    _file = b._file;
    _line = b._line;
    _function = b._function;
    return *this;
}


// --------------------------------------------------------------------------
// demangle
// --------------------------------------------------------------------------

std::string hls::lang::demangle(char const* name) {

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
// exception_t
// --------------------------------------------------------------------------

exception_t::exception_t() {
    _what = "<throwed a exception>";
    _fillstack();
}


exception_t::exception_t(const char* msg) {
    assert(msg);
    _what = msg;
    _fillstack();
}


exception_t::exception_t(const std::string& msg) {
    _what = msg;
    _fillstack();
}


exception_t::~exception_t() throw () {
    _callstack.clear();
}


// --------------------------------------------------------------------------

void exception_t::_fillstack() {
    for(block_t* top = block_t::_top; top != BLOCK_END; top=top->_prev)
        _callstack.push_back(*top);
}


exception_t& exception_t::setcs(const char *fi, int l, const char *fu) {
    block_t here;
    here._file = fi;
    here._line = l;
    here._function = fu;

    _callstack.insert(_callstack.begin(), here);
    return (*this);
}

// --------------------------------------------------------------------------

void exception_t::printstack() {
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

