/**
 * @file
 */
#pragma once

#include "bi/data/PrimitiveValue.hpp"

namespace bi {
/**
 * Conversions from primitive types to C++ built-in types.
 */
template<class T>
struct value_type {
  typedef T type;
};

template<class T, class Group>
struct value_type<bi::PrimitiveValue<T,Group>> {
  typedef T type;
};
}
