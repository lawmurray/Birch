/**
 * @file
 */
#pragma once

#include "libbirch/SwapContext.hpp"

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

/**
 * @def STANDARD_SWAP_CONTEXT
 *
 * When lazy deep clone is in use, swaps into the context of this object.
 */
#if ENABLE_LAZY_DEEP_CLONE
#define STANDARD_SWAP_CONTEXT SwapContext swap(context.get());
#else
#define STANDARD_SWAP_CONTEXT
#endif

/**
 * @def STANDARD_DECLARE_SELF
 *
 * Declare `self` within a member function.
 */
#define STANDARD_DECLARE_SELF Shared<this_type> self(this);

/**
 * @def STANDARD_DECLARE_LOCAL
 *
 * Declare `local` within a member fiber.
 */
#define STANDARD_DECLARE_LOCAL Shared<class_type> local(this);
