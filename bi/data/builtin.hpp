/**
 * @file
 */
#pragma once

#include "bi/data/PrimitiveValue.hpp"
#include "bi/data/StackGroup.hpp"

#include <utility>
#include <cstdint>

namespace bi {
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
 * Built in string type.
 */
template<class Group = StackGroup>
using String = PrimitiveValue<const char*,Group>;

}
