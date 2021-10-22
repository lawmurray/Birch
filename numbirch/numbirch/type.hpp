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

}
