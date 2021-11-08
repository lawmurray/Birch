/**
 * @file
 */
#pragma once

#include "numbirch/macro.hpp"

#include <type_traits>

namespace numbirch {
template<class T, int D> class Array;

/**
 * Is `T` an integral type?
 * 
 * @ingroup numeric
 * 
 * An integral type is one of `int` or `bool`, or an Array of such.
 */
template<class T>
struct is_integral {
  static constexpr bool value = false;
};
template<>
struct is_integral<int> {
  static constexpr bool value = true;
};
template<>
struct is_integral<bool> {
  static constexpr bool value = true;
};
template<class T, int D>
struct is_integral<Array<T,D>> {
  static constexpr bool value = is_integral<T>::value;
};
template<class T>
inline constexpr bool is_integral_v = is_integral<T>::value;

/**
 * Is `T` a floating point type?
 * 
 * @ingroup numeric
 * 
 * A floating point type is one of `double` or `float`, or an Array of such.
 */
template<class T>
struct is_floating_point {
  static constexpr bool value = false;
};
template<>
struct is_floating_point<double> {
  static constexpr bool value = true;
};
template<>
struct is_floating_point<float> {
  static constexpr bool value = true;
};
template<class T, int D>
struct is_floating_point<Array<T,D>> {
  static constexpr bool value = is_floating_point<T>::value;
};
template<class T>
inline constexpr bool is_floating_point_v = is_floating_point<T>::value;

/**
 * Is `T` an arithmetic type?
 * 
 * @ingroup numeric
 * 
 * An arithmetic type is an integral or floating point type.
 * 
 * @see is_integral, is_floating_point
 */
template<class T>
struct is_arithmetic {
  static constexpr bool value = is_integral<T>::value ||
      is_floating_point<T>::value;
};
template<class T>
inline constexpr bool is_arithmetic_v = is_arithmetic<T>::value;

/**
 * Value type of an arithmetic type. For a basic type this is an identity
 * function, for an array type it is the element type.
 * 
 * @ingroup numeric
 */
template<class T>
struct value {
  using type = T;
};
template<class T, int D>
struct value<Array<T,D>> {
  using type = T;
};
template<class T>
using value_t = typename value<T>::type;

/**
 * Dimension of an arithmetic type.
 * 
 * @ingroup numeric
 */
template<class T>
struct dimension {
  static constexpr int value = 0;
};
template<class T, int D>
struct dimension<Array<T,D>> {
  static constexpr int value = D;
};
template<class T>
inline constexpr int dimension_v = dimension<T>::value;

/**
 * Is `T` an array type?
 * 
 * @ingroup numeric
 * 
 * An array type is any arithmetic type with one or more dimensions.
 */
template<class T>
struct is_array {
  static constexpr bool value = is_arithmetic<T>::value &&
      dimension<T>::value > 0;
};
template<class T>
inline constexpr bool is_array_v = is_array<T>::value;

/**
 * Is `T` a scalar type?
 * 
 * @ingroup numeric
 * 
 * A scalar type is any arithmetic type with zero dimensions.
 */
template<class T>
struct is_scalar {
  static constexpr bool value = is_arithmetic<T>::value &&
      dimension<T>::value == 0;
};
template<class T>
inline constexpr bool is_scalar_v = is_scalar<T>::value;

/**
 * Is `T` a basic type?
 * 
 * @ingroup numeric
 * 
 * A basic type is one of `double`, `float`, `int` or `bool`.
 */
template<class T>
struct is_basic {
  static constexpr bool value = false;
};
template<>
struct is_basic<double> {
  static constexpr bool value = true;
};
template<>
struct is_basic<float> {
  static constexpr bool value = true;
};
template<>
struct is_basic<int> {
  static constexpr bool value = true;
};
template<>
struct is_basic<bool> {
  static constexpr bool value = true;
};
template<class T>
inline constexpr bool is_basic_v = is_basic<T>::value;

/**
 * Are two arithmetic types compatible for a binary transform?---Yes, if they
 * have the same number of dimensions, excluding the case where both are a
 * basic. In that latter case the built-in function on host is preferred.
 * 
 * @ingroup numeric
 */
template<class T, class U>
struct is_compatible {
  static constexpr bool value = dimension<T>::value == dimension<U>::value &&
      !(is_basic<T>::value && is_basic<U>::value);
};
template<class T, class U>
inline constexpr bool is_compatible_v = is_compatible<T,U>::value;

/**
 * Promoted arithmetic type for a binary operation.
 * 
 * @ingroup numeric
 * 
 * @tparam T Arithmetic type.
 * @tparam U Arithmetic type.
 */
template<class T, class U>
struct promote {
  using type = void;
};
template<>
struct promote<double,double> {
  using type = double;
};
template<>
struct promote<double,float> {
  using type = double;
};
template<>
struct promote<double,int> {
  using type = double;
};
template<>
struct promote<double,bool> {
  using type = double;
};
template<>
struct promote<float,double> {
  using type = double;
};
template<>
struct promote<float,float> {
  using type = float;
};
template<>
struct promote<float,int> {
  using type = float;
};
template<>
struct promote<float,bool> {
  using type = float;
};
template<>
struct promote<int,double> {
  using type = double;
};
template<>
struct promote<int,float> {
  using type = float;
};
template<>
struct promote<int,int> {
  using type = int;
};
template<>
struct promote<int,bool> {
  using type = int;
};
template<>
struct promote<bool,double> {
  using type = double;
};
template<>
struct promote<bool,float> {
  using type = float;
};
template<>
struct promote<bool,int> {
  using type = int;
};
template<>
struct promote<bool,bool> {
  using type = bool;
};
template<class T, int D, class U, int E>
struct promote<Array<T,D>,Array<U,E>> {
  using type = Array<typename promote<T,U>::type,std::max(D, E)>;
};
template<class T, int D, class U>
struct promote<Array<T,D>,U> {
  using type = Array<typename promote<T,U>::type,D>;
};
template<class T, class U, int E>
struct promote<T,Array<U,E>> {
  using type = Array<typename promote<T,U>::type,E>;
};
template<class T, class U>
using promote_t = typename promote<T,U>::type;

/**
 * Does arithmetic type `T` promote to `U` under promotion rules?
 * 
 * @ingroup numeric
 * 
 * @tparam T Arithmetic type.
 * @tparam U Arithmetic type.
 */
template<class T, class U>
struct promotes_to {
  static constexpr bool value = std::is_same<
      promote_t<T,U>,U>::value;
};
template<class T, class U>
inline constexpr bool promotes_to_v = promotes_to<T,U>::value;

/**
 * Are all argument types integral?
 * 
 * @ingroup numeric
 */
template<class... Args>
struct all_integral {
  //
};
template<class Arg>
struct all_integral<Arg> {
  static const bool value = std::is_integral<
      typename std::decay<Arg>::type>::value;
};
template<class Arg, class... Args>
struct all_integral<Arg,Args...> {
  static const bool value = all_integral<Arg>::value &&
      all_integral<Args...>::value;
};
template<class T, class U>
inline constexpr bool all_integral_v = all_integral<T,U>::value;

/**
 * Pair of values of a given type.
 * 
 * @ingroup numeric
 * 
 * @tparam T Arithmetic type.
 */
template<class T>
struct pair {
  T first, second;
};

/**
 * Triple of values of a given type.
 * 
 * @ingroup numeric
 * 
 * @tparam T Arithmetic type.
 */
template<class T>
struct triple {
  T first, second, third;
};

/**
 * Quad of values of a given type.
 * 
 * @ingroup numeric
 * 
 * @tparam T Arithmetic type.
 */
template<class T>
struct quad {
  T first, second, third, fourth;
};

/**
 * @internal
 *
 * Element of a matrix.
 */
template<class T>
HOST DEVICE T& element(T* x, const int i = 0, const int j = 0,
    const int ld = 0) {
  return x[i + j*ld];
}

/**
 * @internal
 *
 * Element of a matrix.
 */
template<class T>
HOST DEVICE const T& element(const T* x, const int i = 0, const int j = 0,
    const int ld = 0) {
  return x[i + j*ld];
}

/**
 * @internal
 * 
 * Element of a scalar---just returns the scalar.
 */
template<class T>
HOST DEVICE T& element(T& x, const int i = 0, const int j = 0,
    const int ld = 0) {
  return x;
}

/**
 * @internal
 * 
 * Element of a scalar---just returns the scalar.
 */
template<class T>
HOST DEVICE const T& element(const T& x, const int i = 0, const int j = 0,
    const int ld = 0) {
  return x;
}

}
