/**
 * @file
 */
#pragma once

/**
 * @def libbirch_create_function_
 *
 * Defines the standard @c create_() member function required of objects.
 */
#define libbirch_create_function_ \
  template<class... Args> \
  static class_type_* create_(Args&&... args) { \
    return emplace_(libbirch::allocate<sizeof(class_type_)>(), args...); \
  }

/**
 * @def libbirch_emplace_function_
 *
 * Defines the standard @c emplace_() member function required of objects.
 */
#define libbirch_emplace_function_ \
  template<class... Args> \
  static class_type_* emplace_(void* ptr, Args&&... args) { \
    auto o = new (ptr) class_type_(args...); \
    o->Counted::size = sizeof(class_type_); \
    return o; \
  }

/**
 * @def libbirch_clone_function_
 *
 * Defines the standard @c clone_() member function required of objects.
 */
#define libbirch_clone_function_ \
  virtual class_type_* clone_(libbirch::Context* context) const { \
    auto o = emplace_(libbirch::allocate<sizeof(class_type_)>(), *this); \
    o->thaw(context); \
    return o; \
  } \
  virtual class_type_* clone_(void* ptr, libbirch::Context* context) const { \
    auto o = emplace_(ptr, *this); \
    o->thaw(context); \
    return o; \
  }

/**
 * @def libbirch_destroy_function_
 *
 * Defines the standard @c destroy_() member function required of objects.
 */
#define libbirch_destroy_function_ \
  virtual void destroy_() { \
    this->~class_type_(); \
  }

/**
 * @def libbirch_declare_self_
 *
 * Declare `self` within a member function.
 */
#define libbirch_declare_self_ libbirch::Init<this_type_> self(this);

/**
 * @def libbirch_declare_local_
 *
 * Declare `local` within a member fiber.
 */
#define libbirch_declare_local_ libbirch::Init<class_type_> local(this);
