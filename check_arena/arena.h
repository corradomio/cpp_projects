//
// Created by Corrado Mio on 04/04/2020.
//

#ifndef CHECK_ARENA_ARENA_H
#define CHECK_ARENA_ARENA_H

#include <cstddef>

template<typename T>
class arena_ptr;

struct arena_t {
    const int MINALLOC = 4;
    size_t   size;
    size_t   free;
    char*    buffer;

    arena_t(size_t size) {
        this->size = size;
        this->free = size;
        this->buffer = new char[size];
    }

    ~arena_t() {
        delete[] buffer;
        buffer = nullptr;
    }

    template<typename T>
    arena_ptr<T> alloc(size_t count);
};

template<typename T>
class arena_ptr {
    arena_t* arena;
    size_t   offset;
public:
    arena_ptr(): arena(nullptr), offset(0) {}

    arena_ptr(arena_t* arena, const size_t offset)
    : arena(arena),
      offset(offset)
    { }

    arena_ptr(const arena_ptr<T>& other)
    : arena(other.arena),
      offset(other.offset)
    { }

    arena_ptr<T>& operator =(const arena_ptr<T>& other) {
        if (this != &other) {
            this->arena = other.arena;
            this->offset = other.offset;
        }
        return *this;
    }

    T* operator->() const { return  get(); }
    T& operator *() const { return *get(); }

    T* get() const { return (T*)&(arena->buffer[offset]); }
};

template<typename T>
arena_ptr<T> arena_t::alloc(size_t count) {
    size_t allocated = count*sizeof(T);
    if (allocated % MINALLOC != 0) allocated += MINALLOC - allocated%MINALLOC;
    if (allocated > free) throw std::bad_alloc();
    free -= allocated;
    return arena_ptr<T>(this, free);
}

#endif //CHECK_ARENA_ARENA_H
