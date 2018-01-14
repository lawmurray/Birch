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
extern std::shared_ptr<World> fiberWorld;

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
  using type = Any;
};

template<class T>
struct super_type<const T> {
  using type = const typename super_type<T>::type;
};

/**
 * Can type @p T be assigned to from type @p U?
 */
template<class T, class U>
struct is_assignable {
  using T1 = typename std::remove_const<T>::type;
  using U1 = typename std::remove_const<U>::type;

  static const bool value = std::is_assignable<T1,U1>::value &&
      (!std::is_const<U>::value || std::is_const<T>::value);
};

/**
 * Does type @p T have a conversion operator to type @p U?
 */
template<class T, class U>
struct is_convertible {
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
