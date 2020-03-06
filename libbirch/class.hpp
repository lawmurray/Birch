/**
 * @file
 */
#pragma once

/**
 * @def LIBBIRCH_CLASS
 *
 * Boilerplate macro for classes to support lazy deep copy. The first
 * argument is the name of the class; this should exclude any generic type
 * arguments. The second argument is the base class; this should include any
 * generic type arguments. The macro should be placed in the public section
 * of the class.
 *
 * LIBBIRCH_CLASS must be immediately followed by LIBBIRCH_MEMBERS, otherwise
 * the replacement code will have invalid syntax. For example:
 *
 *     class A : public B {
 *     public:
 *       LIBBIRCH_CLASS(A, B)
 *       LIBBIRCH_MEMBERS(x, y, z)
 *     private:
 *       int x, y, z;
 *     };
 *
 * The use of a variadic macro here supports base classes that contain
 * commas without special treatment, e.g.
 *
 *     LIBBIRCH_CLASS(A, B<T,U>)
 */
#define LIBBIRCH_CLASS(Name, Base...) \
  virtual void freeze_(const libbirch::Freezer& v) override { \
    this->accept_(v); \
  } \
  \
  virtual Name* copy_(const libbirch::Copier& v) const override { \
    auto o = new Name(*this); \
    o->setLabel(v.label); \
    o->accept_(v); \
    return o; \
  } \
  \
  virtual Name* recycle_(const libbirch::Recycler& v) override { \
    this->thaw(); \
    this->replaceLabel(v.label); \
    this->accept_(v); \
    return this; \
  } \
  \
  LIBBIRCH_ABSTRACT_CLASS(Name, Base)

/**
 * @def LIBBIRCH_ABSTRACT_CLASS
 *
 * Use in place of LIBBIRCH_CLASS when the containing class is abstract.
 */
#define LIBBIRCH_ABSTRACT_CLASS(Name, Base...) \
  virtual const char* getClassName() const { \
    return #Name; \
  } \
  \
  template<class Visitor> \
  void accept_(const Visitor& v) { \
    Base::accept_(v);

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
 *     public:
 *       LIBBIRCH_CLASS(A, B)
 *       LIBBIRCH_MEMBERS(x, y, z)
 *     private:
 *       int x, y, z;
 *     };
 */
#define LIBBIRCH_MEMBERS(members...) \
    v.visit(members); \
  }

/**
 * @def LIBBIRCH_SELF
 *
 * Boilerplate macro to occur first in a member function or fiber. Declares
 * the local variable `self`, to use in place of the usual `this`.
 */
#define LIBBIRCH_SELF \
  [[maybe_unused]] libbirch::LazyInitPtr<this_type_> self(this, this->getLabel());

#include "libbirch/Freezer.hpp"
#include "libbirch/Copier.hpp"
#include "libbirch/Recycler.hpp"
