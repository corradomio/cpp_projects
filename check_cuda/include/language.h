//
// Created by Corrado Mio on 08/03/2024.
//

// f16, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64,
// f128? i128? u128?

#ifndef LANGUAGE_H
#define LANGUAGE_H

// = delete
// = default
// = nobody

#ifndef interface
#define interface struct
#define nobody 0
#endif

#ifndef self
#define self (*this)
#endif

#ifndef elsif
#define elsif else if
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

#endif //LANGUAGE_H
