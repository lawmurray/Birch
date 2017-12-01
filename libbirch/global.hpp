/**
 * @file
 */
#pragma once

#include <limits>
#include <type_traits>
#include <cstdint>
#include <cstddef>
#include <random>

namespace bi {
class AllocationMap;
class FiberWorld;

/**
 * The allocation map for copy-on-write of objects between worlds.
 */
extern AllocationMap* allocationMap;

/**
 * The world of the currently running fiber.
 */
extern FiberWorld* fiberWorld;

/**
 * Random number generator.
 */
extern std::mt19937_64 rng;

/**
 * Constant to indicate a mutable value. Zero is convenient here, as it
 * enables multiplication to convolve multiple values.
 *
 * @ingroup libbirch
 */
static constexpr size_t mutable_value = 0;

/**
 * The super type of type @p T. Specialised in forward declarations of
 * classes.
 */
template<class T>
struct super_type {
  typedef class Any type;
};

/**
 * Does type @p T have a conversion operator to type @p U?
 */
template<class T, class U>
struct has_conversion {
  /* conversion operators in generated code are marked explicit, they return
   * true for std::is_constructible, but false for std::is_convertible as they
   * cannot partipicate in implicit casts; if they were not marked explicit,
   * std::is_constructible would also use standard conversions before and
   * after the user-defined conversion, which causes ambiguity (e.g. a
   * user-defined conversion to int can be further standard-converted to
   * double, which would allow type @c T to be converted to type double
   * unintentionally */
  static const bool value = std::is_constructible<U,T>::value &&
      !std::is_convertible<T,U>::value;
};
}
