//
// Created by Corrado Mio on 24/03/2024.
//

// #ifndef STDX_TRYEXC_H
// #define STDX_TRYEXC_H
//
// #include <typeinfo>
// #include <exception>
// #include <string>
// #include <vector>
//
// /*
//  * Exception trowing:
//  *
//  *      throw stdx::exception(<message>)
//  *      throw stdx::exception(<message>).from_here()
//  *
//  * Note: 'from_here()' is a MACRO converted in:
//  *
//  *      setcs(__FILE__,__LINE__,__FUNCTION__)
//  *
//  * Supported syntax:
//  *
//  *      _block { ... }
//  *      _block_end
//  *
//  *      { _block_  ... }
//  *
//  */
//
// namespace stdx {
//
//     // ----------------------------------------------------------------------
//     // block_t
//     // ----------------------------------------------------------------------
//
//     class block_t {
//     protected:
//         /**
//          * Usato per mantenere traccia del top dello stack dei try/catch
//          *
//          * Nota: DEVE essere un thread-local
//          */
//         static thread_local block_t* _top;
//
//         /// link al try precedente
//         block_t*    _prev;
//
//         /// nome del file/linea/nome della funzione
//         const char* _file;
//         int         _line;
//         const char* _function;
//
//         /// la classe 'stdx::exception' puo' accedere alle informazioni di
//         /// questa classe
//         friend class exception;
//
//     public:
//         /// costruttore usato nella macro '_try'
//         block_t(const char* fi, int l, const char* fu)
//             : _file(fi), _line(l), _function(fu), _prev(_top)
//         { _top = this; }
//         ~block_t() { if(_prev) _top = _prev; }
//
//         /// costructtori usati nella creazione del callstack
//         block_t(): _file(nullptr), _line(0), _function(nullptr), _prev(nullptr) { }
//         block_t(const block_t& t)
//             : _file(t._file), _line(t._line), _function(t._function), _prev(nullptr) {}
//
//         //        const char* file()     const { return _file; }
//         //        int         line()     const { return _line; }
//         //        const char* function() const { return _function; }
//
//         block_t& operator =(const block_t& t) {
//             _file = t._file;
//             _line = t._line;
//             _function = t._function;
//             return *this;
//         }
//     };
//
//
//     // ----------------------------------------------------------------------
//     // demangle
//     // ----------------------------------------------------------------------
//
//     std::string demangle(const char* name);
//
//     template <class T>
//     std::string typestr(const T& t) {
//         return demangle(typeid(t).name());
//     }
//
//
//     // ----------------------------------------------------------------------
//     // stdx::exception
//     // ----------------------------------------------------------------------
//
//     class exception : std::exception
//     {
//         std::vector<block_t>  _callstack;
//         std::string _what;
//
//         void _fillstack();
//
//     public:
//         exception();
//         exception(const char* msg);
//         exception(const std::string& msg);
//         virtual ~exception() throw ();
//
//         virtual const char* what() const throw() { return _what.c_str(); }
//
//     public:
//         std::vector<block_t>& callstack() const { return (std::vector<block_t>&)_callstack; }
//         void printstack();
//
//     public:
//         exception& setcs(const char *fi, int l, const char *fu);
//     };
//
// };
//
//
// #define  from_here()    setcs(__FILE__,__LINE__,__FUNCTION__)
//
// #define _block          { stdx::block_t binfo(__FILE__,__LINE__,__FUNCTION__);
// #define _block_end      }
// #define _end_block      _block_end
//
// #define _block_         stdx:block_t(__FILE__,__LINE__,__FUNCTION__);
//
// #endif //STDX_TRYEXC_H
