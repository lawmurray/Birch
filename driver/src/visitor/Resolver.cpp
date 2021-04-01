/**
 * @file
 */
#include "src/visitor/Resolver.hpp"

#include "src/exception/all.hpp"
#include "src/visitor/all.hpp"
#include "src/primitive/system.hpp"

birch::Resolver::Resolver(Package* currentPackage, Class* currentClass) :
    ScopedModifier(currentPackage, currentClass) {
  //
}

birch::Resolver::~Resolver() {
  //
}

birch::Expression* birch::Resolver::modify(Parameter* o) {
  scopes.back()->add(o);
  return ScopedModifier::modify(o);
}

birch::Statement* birch::Resolver::modify(LocalVariable* o) {
  scopes.back()->add(o);
  return ScopedModifier::modify(o);
}

birch::Expression* birch::Resolver::modify(NamedExpression* o) {
  ScopedModifier::modify(o);
  if (inMember) {
    /* Clearly a member something, but cannot determine whether this
     * something is a variable or function without type deduction. Instead
     * categorize as MEMBER_UNKNOWN, and let C++ handle the rest */
    o->category = MEMBER_UNKNOWN;
  } else if (inGlobal) {
    /* just check the global scope */
    scopes.front()->lookup(o);
  } else {
    for (auto iter = scopes.rbegin(); iter != scopes.rend() && !o->category;
        ++iter) {
      (*iter)->lookup(o);
    }
    if (!o->category) {
      if (currentClass) {
        /* While a local or global should be resolved, a member may not be
         * if it belongs to a base class. This is particularly the case when
         * the "curiously recurring template pattern" (CRTP) is used, where
         * the base class is given by a generic type parameter, and the
         * particular type is unknown at this point (and won't be known,
         * without type deduction). In this case, categorize as
         * MEMBER_UNKNOWN, and let the C++ handle the rest. */
        o->category = MEMBER_UNKNOWN;
      } else {
        throw UndefinedException(o);
      }
    }
  }
  return o;
}

birch::Type* birch::Resolver::modify(NamedType* o) {
  ScopedModifier::modify(o);
  for (auto iter = scopes.rbegin(); iter != scopes.rend() && !o->category;
      ++iter) {
    (*iter)->lookup(o);
  }
  if (!o->category) {
    o->category = GENERIC_TYPE;
  }
  return o;
}

birch::Statement* birch::Resolver::modify(Class* o) {
  scopes.back()->inherit(o);
  return ScopedModifier::modify(o);
}

birch::Statement* birch::Resolver::modify(Function* o) {
  if (o->returnType->isTypeOf() && !o->isGeneric() &&
      (currentClass && !currentClass->isGeneric())) {
    throw ReturnTypeDeductionException(o);
  }
  return ScopedModifier::modify(o);
}

birch::Statement* birch::Resolver::modify(MemberFunction* o) {
  if (o->returnType->isTypeOf() && !o->isGeneric() &&
      (currentClass && !currentClass->isGeneric())) {
    throw ReturnTypeDeductionException(o);
  }
  return ScopedModifier::modify(o);
}
