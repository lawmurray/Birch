/**
 * @file
 */
#pragma once

#include <type_traits>

namespace numbirch {
/**
 * Promoted arithmetic type of an operation between values of given arithmetic
 * types. For example, `typename promote<int,double>::type` is `double`. This
 * is defined by the numerical promotion rules of C++.
 * 
 * @tparam T Arithmetic type.
 * @tparam U Arithmetic type.
 */
template<class T, class U, std::enable_if_t<std::is_arithmetic<T>::value &&
    std::is_arithmetic<U>::value,int> = 0>
struct promote {
  using type = decltype(T()*U());
};

/**
 * Does arithmetic type `T` promote to `U` under promotion rules?
 * 
 * @tparam T Arithmetic type.
 * @tparam U Arithmetic type.
 */
template<class T, class U, std::enable_if_t<std::is_arithmetic<T>::value &&
    std::is_arithmetic<U>::value,int> = 0>
struct promotes_to {
  static constexpr bool value = std::is_same<
      typename promote<T,U>::type,U>::value;
};

/**
 * Pair of values of a given type.
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
 * @tparam T Arithmetic type.
 */
template<class T>
struct triple {
  T first, second, third;
};

/**
 * Quad of values of a given type.
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
 * Are all argument types integral?
 * 
 * @ingroup array
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

}
