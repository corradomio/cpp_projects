//
// Created by Corrado Mio on 25/09/2020.
//
#include <stdio.h>
#include <mem/ref_count.h>

struct C {
    static int gid;
    int id;

    C():id(++gid) { printf("+C(%d)\n", id); }
    ~C() { printf("-C(%d)\n", id); }
};

struct D : public C {
    D():C() { printf("+D(%d)\n", id); }
    ~D() { printf("-D(%d)\n", id); }
};

int C::gid = 0;

int main5() {

    //C* pc = new D;
    //D* pd = static_cast<D*>(pc);
    //
    //mem::ref_ptr<C> p(new D);
    //mem::ref_ptr<C> q = mem::ref_ptr<C>(new C);
    //mem::ref_ptr<D> r = mem::ref_cast<D>(p);

    return 0;
}
