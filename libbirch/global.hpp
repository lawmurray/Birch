/**
 * @file
 */
#pragma once

#include <type_traits>
#include <random>
#include <memory>
#include <cstdint>
#include <cstddef>

namespace bi {
class World;
class Any;
template<class T> class SharedPointer;
template<class T> class WeakPointer;

/**
 * The world of the currently running fiber.
 */
extern World* fiberWorld;

/**
 * Flag set when an object is being cloned.
 */
extern bool fiberClone;

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
static constexpr int64_t mutable_value = 0;

/**
 * The super type of type @p T. Specialised in forward declarations of
 * classes.
 */
template<class T>
struct super_type {
  using type = Any;
};

template<class T>
struct super_type<const T> {
  using type = const typename super_type<T>::type;
};

/**
 * Does type @p T hs an assignment operator for type @p U?
 */
template<class T, class U>
struct has_assignment {
  static const bool value = has_assignment<typename super_type<T>::type,U>::value;
};
template<class U>
struct has_assignment<Any,U> {
  static const bool value = false;
};

/**
 * Does type @p T have a conversion operator for type @p U?
 */
template<class T, class U>
struct has_conversion {
  static const bool value = has_conversion<typename super_type<T>::type,U>::value;
};
template<class U>
struct has_conversion<Any,U> {
  static const bool value = false;
};
}
