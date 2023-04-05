/*
 * File:   trycatch.hpp
 * Author: ...
 *
 * Created on April 24, 2013, 10:45 AM
 */

#ifndef HLS_LANG_TRYCATCH_HPP
#define HLS_LANG_TRYCATCH_HPP

#include <string>
#include <vector>
#include "exception.h"

/**
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
        try_t(const char* fi, int l, const char* fu);
       ~try_t();
        /// costruttori usati nella creazione del callstack
        try_t();
        try_t(const block_t& t);

        void finalize();
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


#endif   /* HLS_LANG_TRYCATCH_HPP */
