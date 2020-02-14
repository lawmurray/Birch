/**
 * @file
 */
#pragma once

/**
 * @def LIBBIRCH_CLASS
 *
 * Boilerplate macro to declare member functions necessary for lazy deep
 * copy support. The first argument gives the class, the second its base
 * class, the remaining the member variables of the class. It is recommended
 * that all member variables are included, although it is actually  possible
 * to omit those with a type that does not include a class eligible for lazy
 * deep copy.
 */
#define LIBBIRCH_CLASS(Class, Base, ...) \
  virtual Class* clone_() const { \
    return new Class(*this); \
  } \
  \
  template<class V_> \
  void accept_(V_& v_) { \
    Base::accept_(v_); \
    v_.visit(__VA_ARGS__); \
  }

/**
 * @def LIBBIRCH_ABSTRACT_CLASS
 *
 * As LIBBIRCH_CLASS, but for an abstract class.
 */
#define LIBBIRCH_ABSTRACT_CLASS(Class, Base, ...) \
  template<class V_> \
  void accept_(V_& v_) { \
    Base::accept_(v_); \
    v_.visit(__VA_ARGS__); \
  }

/**
 * @def LIBBIRCH_CLASS_NAME
 */
#define LIBBIRCH_CLASS_NAME(ClassName) \
  virtual const char* getClassName() const {\
    return ClassName; \
  }

/**
 * @def LIBBIRCH_SELF
 *
 * Boilerplate macro to occur first in a member function or fiber. Declares
 * the local variable `self`, to use in place of the usual `this`.
 */
#define LIBBIRCH_SELF \
  [[maybe_unused]] libbirch::LazyInitPtr<this_type_> self(this, this->getLabel());

namespace bi {
  namespace type {
/**
 * Super type of another. This is specialized for all classes that are
 * derived from Any to indicate their super type without having to
 * instantiate that type.
 */
template<class T>
struct super_type {
  //
};
  }
}
