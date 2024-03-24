//
// Created by Corrado Mio on 08/03/2024.
//

#ifndef STDX_LANGUAGE_H
#define STDX_LANGUAGE_H

#ifndef self
#define self (*this)
#endif

#ifndef elsif
#define elsif else if
#endif

#ifndef elif
#define elif else if
#endif

// #ifndef raise
// #define raise throw
// #endif


/// cast the object of type 'T' to 'T::super&'
/// where 'super' is defined as
///
///     class D : public A {
///     public:
///         using super = A;
///         ...
///     }
///
/// \tparam T
/// \param object
/// \return
template<typename T>
typename T::super& up_cast(T& elem) {
    return reinterpret_cast<typename T::super&>(elem);
}


#endif //STDX_LANGUAGE_H
