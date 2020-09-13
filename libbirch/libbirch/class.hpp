/**
 * @file
 */
#pragma once

/**
 * @internal
 *
 * @def LIBBIRCH_BASE
 *
 * Declare the base type for a class.
 */
#define LIBBIRCH_BASE(Base...) \
  using base_type_ = Base; \

/**
 * @internal
 *
 * @def LIBBIRCH_COMMON
 *
 * Declare common functions for classes.
 */
#define LIBBIRCH_COMMON(Name, Base...) \
  virtual const char* getClassName() const { \
    return #Name; \
  } \
  \
  virtual unsigned size_() const override { \
    return sizeof(*this); \
  } \
  \
  virtual void finish_(libbirch::Label* label) override { \
    this->accept_(libbirch::Finisher(label)); \
  } \
  \
  virtual void freeze_() override { \
    this->accept_(libbirch::Freezer()); \
  } \
  \
  virtual Name* copy_(libbirch::Label* label) const override { \
    auto src = static_cast<const void*>(this); \
    auto dst = libbirch::allocate(sizeof(*this)); \
    std::memcpy(dst, src, sizeof(*this)); \
    auto o = static_cast<Name*>(dst); \
    o->accept_(libbirch::Copier(label)); \
    return o; \
  } \
  \
  virtual void recycle_(libbirch::Label* label) override { \
    this->accept_(libbirch::Recycler(label)); \
  } \
  \
  virtual void mark_() override { \
    this->accept_(libbirch::Marker()); \
  } \
  \
  virtual void scan_() override { \
    this->accept_(libbirch::Scanner()); \
  } \
  \
  virtual void reach_() override { \
    this->accept_(libbirch::Reacher()); \
  } \
  \
  virtual void collect_() override { \
    this->accept_(libbirch::Collector()); \
  }

/**
 * @def LIBBIRCH_CLASS
 *
 * Boilerplate macro for classes to support lazy deep copy. The first
 * argument is the name of the class; this should exclude any generic type
 * arguments. The second argument is the base class; this should include any
 * generic type arguments. It is recommended that the macro is used at the
 * very end of the class definition.
 *
 * LIBBIRCH_CLASS must be immediately followed by LIBBIRCH_MEMBERS, otherwise
 * the replacement code will have invalid syntax. For example:
 *
 *     class A : public B {
 *     private:
 *       int x, y, z;
 *
 *       LIBBIRCH_CLASS(A, B)
 *       LIBBIRCH_MEMBERS(x, y, z)
 *     };
 *
 * The use of a variadic macro here supports base classes that contain
 * commas without special treatment, e.g.
 *
 *     LIBBIRCH_CLASS(A, B<T,U>)
 */
#define LIBBIRCH_CLASS(Name, Base...) public: \
  LIBBIRCH_BASE(Base) \
  LIBBIRCH_COMMON(Name, Base) \
   \
  auto shared_from_this_() { \
    return libbirch::Lazy<libbirch::Shared<Name>>(this); \
  } \
  \
  template<class Visitor> \
  void accept_(const Visitor& v_) { \
    base_type_::accept_(v_);

/**
 * @def LIBBIRCH_ABSTRACT_CLASS
 *
 * Use in place of LIBBIRCH_CLASS when the containing class is abstract.
 */
#define LIBBIRCH_ABSTRACT_CLASS(Name, Base...) public: \
  LIBBIRCH_BASE(Base) \
   \
  auto shared_from_this_() { \
    return libbirch::Lazy<libbirch::Shared<Name>>(this); \
  } \
  \
  template<class Visitor> \
  void accept_(const Visitor& v_) { \
    base_type_::accept_(v_);

/**
 * @def LIBBIRCH_MEMBERS
 *
 * Boilerplate macro for classes to support lazy deep copy. The arguments
 * list all member variables of the class (and should not include member
 * variables of base classes---these should have their own LIBBIRCH_MEMBERS
 * macro use.
 *
 * LIBBIRCH_MEMBERS must be immediately preceded by LIBBIRCH_CLASS or
 * LIBBIRCH_ABSTRACT_CLASS, otherwise the replacement code will have invalid
 * syntax. For example:
 *
 *     class A : public B {
 *     private:
 *       int x, y, z;
 *
 *       LIBBIRCH_CLASS(A, B)
 *       LIBBIRCH_MEMBERS(x, y, z)
 *     };
 */
#define LIBBIRCH_MEMBERS(members...) \
    v_.visit(members); \
  } \
  using member_type_ = libbirch::Tuple<typename base_type_::member_type_,decltype(libbirch::make_tuple(members))>;

#include "libbirch/Finisher.hpp"
#include "libbirch/Freezer.hpp"
#include "libbirch/Copier.hpp"
#include "libbirch/Recycler.hpp"
#include "libbirch/Marker.hpp"
#include "libbirch/Scanner.hpp"
#include "libbirch/Reacher.hpp"
#include "libbirch/Collector.hpp"
