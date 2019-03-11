/**
 * @file
 */
#pragma once

namespace libbirch {
/**
 * The super type of type @p T. Specialised in forward declarations of
 * classes.
 */
template<class T>
struct super_type {
  using type = void;
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
struct has_assignment<void,U> {
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
struct has_conversion<void,U> {
  static const bool value = false;
};
}

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
  virtual class_type_* clone_() const { \
    return emplace_(libbirch::allocate<sizeof(class_type_)>(), *this); \
  } \
  virtual class_type_* clone_(void* ptr) const { \
    return emplace_(ptr, *this); \
  }

/**
 * @def libbirch_destroy_function_
 *
 * Defines the standard @c destroy_() member function required of objects.
 */
#define libbirch_destroy_function_ \
  virtual void destroy_() override { \
    this->~class_type_(); \
  }

/**
 * @def libbirch_swap_context_
 *
 * When lazy deep clone is in use, swaps into the context of this object.
 */
#if ENABLE_LAZY_DEEP_CLONE
#define libbirch_swap_context_ libbirch::SwapContext swap_(this);
#else
#define libbirch_swap_context_
#endif

/**
 * @def libbirch_declare_self_
 *
 * Declare `self` within a member function.
 */
#define libbirch_declare_self_ libbirch::Shared<this_type_> self(this);

/**
 * @def libbirch_declare_local_
 *
 * Declare `local` within a member fiber.
 */
#define libbirch_declare_local_ libbirch::Shared<class_type_> local(this);
