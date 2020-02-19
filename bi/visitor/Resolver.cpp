/**
 * @file
 */
#include "bi/visitor/Resolver.hpp"

#include "bi/exception/all.hpp"
#include "bi/visitor/all.hpp"

bi::Resolver::Resolver(Package* currentPackage, Class* currentClass,
    Fiber* currentFiber) :
    ScopedModifier(currentPackage, currentClass, currentFiber) {
  //
}

bi::Resolver::~Resolver() {
  //
}

bi::Expression* bi::Resolver::modify(Parameter* o) {
  scopes.back()->add(o);
  return ScopedModifier::modify(o);
}

bi::Statement* bi::Resolver::modify(LocalVariable* o) {
  scopes.back()->add(o);
  return ScopedModifier::modify(o);
}

bi::Expression* bi::Resolver::modify(NamedExpression* o) {
  ScopedModifier::modify(o);
  if (inMember) {
    /* Clearly a member something, but cannot determine whether this
     * something is a variable, function or fiber without type deduction.
     * Instead categorize as MEMBER_UNKNOWN, and let C++ handle the rest */
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
        assert(false);
        throw UnresolvedException(o);
      }
    }
  }
  return o;
}

bi::Type* bi::Resolver::modify(NamedType* o) {
  ScopedModifier::modify(o);
  for (auto iter = scopes.rbegin(); iter != scopes.rend() && !o->category;
      ++iter) {
    (*iter)->lookup(o);
  }
  if (!o->category) {
    throw UnresolvedException(o);
  }
  return o;
}

bi::Statement* bi::Resolver::modify(Class* o) {
  scopes.back()->inherit(o);
  return ScopedModifier::modify(o);
}

bi::Statement* bi::Resolver::modify(Fiber* o) {
  ScopedModifier::modify(o);

  /* construct the start function */
  Resumer resumer;
  auto typeParams = o->typeParams->accept(&cloner);
  auto params = o->params->accept(&cloner);
  auto returnType = o->returnType->accept(&cloner);
  auto braces = o->braces->accept(&resumer);
  auto start = new Function(NONE, o->name, typeParams, params, returnType,
      braces, o->loc);
  start->number = 0;
  o->start = start;

  /* resolve; use a new Resolver here to ensure outer scopes correct */
  Resolver resolver(currentPackage, currentClass);
  o->start = o->start->accept(&resolver);

  return o;
}

bi::Statement* bi::Resolver::modify(MemberFiber* o) {
  ScopedModifier::modify(o);

  /* construct the start function */
  Resumer resumer;
  auto typeParams = o->typeParams->accept(&cloner);
  auto params = o->params->accept(&cloner);
  auto returnType = o->returnType->accept(&cloner);
  auto braces = o->braces->accept(&resumer);
  auto start = new MemberFunction(NONE, o->name, typeParams, params,
      returnType, braces, o->loc);
  start->number = 0;
  o->start = start;

  /* resolve; use a new Resolver here to ensure outer scopes correct */
  Resolver resolver(currentPackage, currentClass);
  o->start = o->start->accept(&resolver);

  return o;
}

bi::Statement* bi::Resolver::modify(Yield* o) {
  ScopedModifier::modify(o);

  /* determine the parameters for the resume function, these being the
   * original parameters plus any local variables in scope at the yield, all
   * of which must be restored when resuming execution */
  Expression* params = nullptr;
  for (auto iter = scopes.rbegin(); iter != scopes.rend(); ++iter) {
    auto scope = *iter;
    params = addParameters((*iter)->parameters, params);
    params = addParameters((*iter)->localVariables, params);
  }
  if (!params) {
    params = new EmptyExpression(o->loc);
  }

  if (currentFiber) {
    /* construct the complete resume function */
    Resumer resumer(o);
    auto typeParams = currentFiber->typeParams->accept(&cloner);
    auto returnType = currentFiber->returnType->accept(&cloner);
    auto braces = currentFiber->braces->accept(&resumer);
    auto resume = new Function(NONE, currentFiber->name, typeParams, params,
        returnType, braces, o->loc);
    resume->number = o->number;
    o->resume = resume;
  } else {
    /* already within the construction of a resume function, just create
     * a function signature so that params are available */
    auto typeParams = new EmptyExpression(o->loc);
    auto returnType = new EmptyType(o->loc);
    auto braces = new EmptyStatement(o->loc);
    auto resume = new Function(NONE, new Name(), typeParams, params,
        returnType, braces, o->loc);
    resume->number = o->number;
    o->resume = resume;
  }

  /* resolve; use a new Resolver here to ensure outer scopes correct */
  Resolver resolver(currentPackage, currentClass);
  o->resume = o->resume->accept(&resolver);

  return o;
}
