/**
 * @file
 */
#pragma once

#include "libbirch/config.hpp"

namespace bi {
class Any;

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
  static const bool value =
      has_assignment<typename super_type<T>::type,U>::value;
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
  static const bool value =
      has_conversion<typename super_type<T>::type,U>::value;
};
template<class U>
struct has_conversion<Any,U> {
  static const bool value = false;
};
}

/**
 * @def STANDARD_CREATE_FUNCTION
 *
 * Defines the standard @c create() member function required of objects.
 */
#define STANDARD_CREATE_FUNCTION \
  template<class... Args> \
  static class_type* create(Args&&... args) { \
    return emplace(allocate<sizeof(class_type)>(), args...); \
  }

/**
 * @def STANDARD_EMPLACE_FUNCTION
 *
 * Defines the standard @c emplace() member function required of objects.
 */
#define STANDARD_EMPLACE_FUNCTION \
  template<class... Args> \
  static class_type* emplace(void* ptr, Args&&... args) { \
    auto o = new (ptr) class_type(args...); \
    o->size = sizeof(class_type); \
    return o; \
  }

/**
 * @def STANDARD_CLONE_FUNCTION
 *
 * Defines the standard @c clone() member function required of objects.
 */
#define STANDARD_CLONE_FUNCTION \
  virtual class_type* clone() const { \
    return emplace(allocate<sizeof(class_type)>(), *this); \
  } \
  virtual class_type* clone(void* ptr) const { \
    return emplace(ptr, *this); \
  }

/**
 * @def STANDARD_DESTROY_FUNCTION
 *
 * Defines the standard @c destroy() member function required of objects.
 */
#define STANDARD_DESTROY_FUNCTION \
  virtual void destroy() override { \
    this->~class_type(); \
  }
