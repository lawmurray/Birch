/**
 * @file
 */
#pragma once

#include "numbirch/array/Array.hpp"
#include "numbirch/array/Scalar.hpp"

#include <type_traits>

namespace numbirch {
/**
 * @internal
 * 
 * Is `T` an array type?
 */
template<class T>
struct is_array {
  static const bool value = false;
};

template<class T, int D>
struct is_array<Array<T,D>> {
  static const bool value = true;
};

/**
 * A future. Array objects already act as futures, so `Future<T> == T` for
 * those. For all others `Future<T> == Scalar<T>`, wrapping the type in an
 * array.
 * 
 * @ingroup array
 */
template<class T>
using Future = typename std::conditional<is_array<T>::value,T,
    Scalar<T>>::type;

///@todo Deduction guides for type aliases require C++20
// template<class T>
// Future(const T&) -> Future<T>;
// template<class T, int D>
// Future(const Array<T,D>&) -> Future<Array<T,D>>;

}
