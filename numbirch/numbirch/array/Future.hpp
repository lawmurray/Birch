/**
 * @file
 */
#pragma once

#include "numbirch/array.hpp"
#include "numbirch/utility.hpp"

namespace numbirch {
/**
 * A future. Array objects are already futures, so `Future<T> == T` for those.
 * For all others `Future<T> == Array<T,0>`, wrapping the type in an array.
 * 
 * @ingroup array
 */
template<class T>
using Future = std::conditional_t<is_array_v<T>,T,Array<T,0>>;

}
