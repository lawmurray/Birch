/**
 * @file
 */
#pragma once

#include "bi/data/PrimitiveValue.hpp"

#include "bi/data/MemoryGroup.hpp"

#include <utility>
#include <cstdint>
#include <functional>

namespace bi {
  namespace model {
/**
 * Built-in boolean type.
 */
template<class Group = MemoryGroup>
using Boolean = PrimitiveValue<unsigned char,Group>;
// ^ unsigned char simpler to write to NetCDF than bool

/**
 * Built-in 64-bit integer type.
 */
template<class Group = MemoryGroup>
using Integer64 = PrimitiveValue<int64_t,Group>;

/**
 * Built-in 32-bit integer type.
 */
template<class Group = MemoryGroup>
using Integer32 = PrimitiveValue<int32_t,Group>;

/**
 * Built-in double-precision floating-point type.
 */
template<class Group = MemoryGroup>
using Real64 = PrimitiveValue<double,Group>;

/**
 * Built-in single-precision floating-point type.
 */
template<class Group = MemoryGroup>
using Real32 = PrimitiveValue<float,Group>;

/**
 * Built-in string type.
 */
template<class Group = MemoryGroup>
using String = PrimitiveValue<const char*,Group>;
  }
}

namespace bi {
/**
 * Conversions from primitive types to C++ built-in types.
 */
template<class T>
struct value_type {
  typedef T type;
};

template<class Group>
struct value_type<bi::model::Boolean<Group>> {
  typedef typename bi::model::Boolean<Group>::value_type type;
};

template<class Group>
struct value_type<bi::model::Integer64<Group>> {
  typedef typename bi::model::Integer64<Group>::value_type type;
};

template<class Group>
struct value_type<bi::model::Integer32<Group>> {
  typedef typename bi::model::Integer32<Group>::value_type type;
};

template<class Group>
struct value_type<bi::model::Real64<Group>> {
  typedef typename bi::model::Real64<Group>::value_type type;
};

template<class Group>
struct value_type<bi::model::Real32<Group>> {
  typedef typename bi::model::Real32<Group>::value_type type;
};

template<class Group>
struct value_type<bi::model::String<Group>> {
  typedef typename bi::model::String<Group>::value_type type;
};
}
