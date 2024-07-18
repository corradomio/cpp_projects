/*
 * File:   trycatch.hpp
 * Author: ...
 *
 * Created on April 24, 2013, 10:45 AM
 */

#ifndef STDX_TRYCATCH_HPP
#define STDX_TRYCATCH_HPP

#include <typeinfo>
#include <exception>
#include <string>
#include <vector>
// #include "tryexc.h"


/*
 * Exception trowing:
 *
 *      throw stdx::exception(<message>)
 *      throw stdx::exception(<message>).from_here()
 *
 * Note: 'from_here()' is a MACRO converted in:
 *
 *      setcs(__FILE__,__LINE__,__FUNCTION__)
 *
 * Supported syntax:
 *
 *      _block { ... }
 *      _block_end
 *
 *      { _block_  ... }
 *
 */

namespace stdx {

    // ----------------------------------------------------------------------
    // block_t
    // ----------------------------------------------------------------------

    class block_t {
    protected:
        /**
         * Usato per mantenere traccia del top dello stack dei try/catch
         *
         * Nota: DEVE essere un thread-local
         */
        static thread_local block_t* _top;

        /// link al try precedente
        block_t*    _prev;

        /// nome del file/linea/nome della funzione
        const char* _file;
        int         _line;
        const char* _function;

        /// la classe 'stdx::exception' puo' accedere alle informazioni di
        /// questa classe
        friend class exception;

    public:
        /// costruttore usato nella macro '_try'
        block_t(const char* fi, int l, const char* fu)
            : _file(fi), _line(l), _function(fu), _prev(_top)
        { _top = this; }
        ~block_t() { if(_prev) _top = _prev; }

        /// costructtori usati nella creazione del callstack
        block_t(): _file(nullptr), _line(0), _function(nullptr), _prev(nullptr) { }
        block_t(const block_t& t)
            : _file(t._file), _line(t._line), _function(t._function), _prev(nullptr) {}

        //        const char* file()     const { return _file; }
        //        int         line()     const { return _line; }
        //        const char* function() const { return _function; }

        block_t& operator =(const block_t& t) {
            _file = t._file;
            _line = t._line;
            _function = t._function;
            return *this;
        }
    };


    // ----------------------------------------------------------------------
    // demangle
    // ----------------------------------------------------------------------

    std::string demangle(const char* name);

    template <class T>
    std::string typestr(const T& t) {
        return demangle(typeid(t).name());
    }


    // ----------------------------------------------------------------------
    // stdx::exception
    // ----------------------------------------------------------------------

    class exception : std::exception
    {
        std::vector<block_t>  _callstack;
        std::string _what;

        void _fillstack();

    public:
        exception();
        exception(const char* msg);
        exception(const std::string& msg);
        virtual ~exception() throw ();

        virtual const char* what() const throw() { return _what.c_str(); }

    public:
        std::vector<block_t>& callstack() const { return (std::vector<block_t>&)_callstack; }
        void printstack();

    public:
        exception& setcs(const char *fi, int l, const char *fu);
    };

};


#define  from_here()    setcs(__FILE__,__LINE__,__FUNCTION__)

#define _block          { stdx::block_t binfo(__FILE__,__LINE__,__FUNCTION__);
#define _block_end      }
#define _end_block      _block_end

#define _block_         stdx:block_t(__FILE__,__LINE__,__FUNCTION__);


/*
 * Supported syntax:
 *
 *      _try { ... }
 *      _try_end
 *
 *      _try { ... }
 *      _catch(e) { ... }
 *      _catch_all { ... }
 *      _catch_end
 *
 *      _try { ... }
 *      _finally { ... }
 *      _finally_end
 *
 *      _block { ... }
 *      _block_end
 *
 *      {
 *          __block__
 *          ...
 *      }
 *
 * NON supportato:
 *
 *      _try { ... }
 *      _catch(e) { ... }
 *      _catch_all { ... }
 *      _finally { ... }
 *      _finally_end
 *
 * sostituire con:
 *
 *      _try {
 *          _try { ... }
 *          _catch_all { ... }
 *          _catch_end
 *      }
 *      _finally { ... }
 *      _finally_end
 *
 *  Nota:
 *      le parentesi non sono strettamente necessarie!
 */

namespace stdx {

    // ----------------------------------------------------------------------
    // try_t
    // ----------------------------------------------------------------------

    class try_t : public block_t {
        /// se l'eccezione e' stata sollevata per forzare la 'finalize'
        bool _is_finally;

    public:
        /// costruttore usato nella macro '_try'
        try_t(const char* fi, int l, const char* fu): block_t(fi, l, fu), _is_finally(false) {}
        /// costruttori usati nella creazione del callstack
        try_t() : block_t(), _is_finally(false) {}
        try_t(const block_t& t): block_t(t),_is_finally(false) {}

        /// distruttore (ripristina il top dello stack)
        ~try_t() { }

        void finalize() { _is_finally = true; throw (*this); }

        bool is_finally() const { return _is_finally; }
    };
};


#define _try        { stdx::try_t tinfo (__FILE__,__LINE__,__FUNCTION__); try {
#define _try_end    } catch(...) { }}
#define _end_try     _try_end


#define _catch(e)   } catch(e)  {
#define _catch_all  } catch(...) {
#define _catch_end  }}
#define _end_catch   _catch_end


#define _finally     tinfo.finalize(); } catch(...) {
#define _finally_end if (!tinfo.is_finally()) throw; }}
#define _end_finally _finally_end


#define _throw(e)       { throw (( e).setcs(__FILE__,__LINE__,__FUNCTION__)); }
#define _throw_ptr(e)   { throw ((*e).setcs(__FILE__,__LINE__,__FUNCTION__)); }


// NON SI PUO' FARE
// Non si puo' implemenare '_finalize

//#define _TRY_CONCAT_INNER(a, b) a ## b
//#define _TRY_CONCAT(a, b) _TRY_CONCAT_INNER(a, b)
//#define _TRY_NAME(base) _TRY_CONCAT(base, __LINE__)
//
//#define _try        stdx::try_t _TRY_NAME(tinfo) (__FILE__,__LINE__,__FUNCTION__); try
//#define _try_end    catch(...) { }
//#define _end_try     _try_end
//
//#define _catch(e)   catch(e)
//#define _catch_all  catch(...)
//#define _catch_end
//#define _end_catch   _catch_end
//
//#define _finally     tinfo.finalize(); } catch(...)
//#define _finally_end if (!tinfo.is_finally()) throw; }
//#define _end_finally _finally_end


#endif//STDX_TRYCATCH_HPP
