/**
 * @file
 */
#pragma once

#include "bi/data/PrimitiveValue.hpp"
#include "bi/data/StackGroup.hpp"

#include <utility>
#include <cstdint>
#include <functional>

namespace bi {
  namespace model {
/**
 * Built-in boolean type.
 */
template<class Group = StackGroup>
using Boolean = PrimitiveValue<unsigned char,Group>;
// ^ unsigned char simpler to write to NetCDF than bool

/**
 * Built-in 64-bit integer type.
 */
template<class Group = StackGroup>
using Integer64 = PrimitiveValue<int64_t,Group>;

/**
 * Built-in 32-bit integer type.
 */
template<class Group = StackGroup>
using Integer32 = PrimitiveValue<int32_t,Group>;

/**
 * Built-in double-precision floating-point type.
 */
template<class Group = StackGroup>
using Real64 = PrimitiveValue<double,Group>;

/**
 * Built-in single-precision floating-point type.
 */
template<class Group = StackGroup>
using Real32 = PrimitiveValue<float,Group>;

/**
 * Built-in string type.
 */
template<class Group = StackGroup>
using String = PrimitiveValue<const char*,Group>;

/**
 * Built-in lambda type.
 */
template<class Group = StackGroup>
using Lambda = PrimitiveValue<std::function<void()>,Group>;

  }
}

namespace bi {
/**
 * Convenience function for literals.
 */
inline bi::model::Boolean<> make_bool(const unsigned char x) {
  return bi::model::Boolean<>(x);
}

/**
 * Convenience function for literals.
 */
inline bi::model::Integer64<> make_int(const int64_t x) {
  return bi::model::Integer64<>(x);
}

/**
 * Convenience function for literals.
 */
inline bi::model::Real64<> make_real(const double x) {
  return bi::model::Real64<>(x);
}

/**
 * Convenience function for literals.
 */
inline bi::model::String<> make_string(const char* x) {
  return bi::model::String<>(x);
}

}
