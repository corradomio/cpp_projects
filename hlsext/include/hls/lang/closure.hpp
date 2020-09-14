//
// Created by Corrado Mio on 24/05/2015.
//

#ifndef TEST_CLOSURE_HPP
#define TEST_CLOSURE_HPP

namespace hls {
namespace lang {

    template<typename R, typename... A>
    class closure_f {
    protected:
        typedef R (*function)(A...);

        function fun;

        R call(A... args) { return (*fun)(args...); }

    public:
        closure_f() { }

        closure_f(function f) : fun(f) { }

        closure_f(const closure_f &c) : fun(c.fun) { }
    };


    template<typename C, typename D, typename R, typename... A>
    class closure_t : public closure_f<R, A...> {
        typedef C *pointer;
        typedef R (D::*method)(A...);
        typedef closure_f<R, A...> function_closure;
        typedef typename function_closure::function function;

        pointer ptr;
        method mthd;

    public:
        closure_t(function f) : function_closure(f) {
            ptr = reinterpret_cast<pointer>(this);
            mthd = closure_f<R, A...>::call;
        }

    public:
        closure_t(pointer p, method m) : ptr(p), mthd(m) { }
        closure_t(const closure_t &c) : ptr(c.ptr), mthd(c.mthd) { }

        R operator()(A... args) const {
            return (ptr->*mthd)(args...);
        }
    };

    template<typename C, typename D, typename R, typename... A>
    closure_t<C, D, R, A...>
    closureof(C *p, R (D::*m)(A...))
    {
        return closure_t<C, D, R, A...>(p, m);
    }

    template<typename R, typename... A>
    closure_t<closure_f<R, A...>, closure_f<R, A...>, R, A...>
    closureof(R (*f)(A...))
    {
        return closure_t<closure_f<R, A...>, closure_f<R, A...>, R, A...>(f);
    }

}}

#endif //TEST_CLOSURE_HPP
