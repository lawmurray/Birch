/**
 * @file
 */
#pragma once

#include "numbirch/array/Array.hpp"

namespace numbirch {
/**
 * Scalar (zero-dimensional array).
 * 
 * @ingroup array
 */
template<class T>
using Scalar = Array<T,0>;
}
