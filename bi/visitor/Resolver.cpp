/**
 * @file
 */
#include "bi/visitor/Resolver.hpp"

#include "bi/exception/all.hpp"

bi::Resolver::Resolver() {
  //
}

bi::Resolver::~Resolver() {
  //
}

bi::Expression* bi::Resolver::modify(NamedExpression* o) {
  Modifier::modify(o);
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
      if (inClass) {
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
  Modifier::modify(o);
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
  o->yield = o->yield->accept(this);
  return ScopedModifier::modify(o);
}

bi::Statement* bi::Resolver::modify(MemberFiber* o) {
  o->yield = o->yield->accept(this);
  return ScopedModifier::modify(o);
}

bi::Statement* bi::Resolver::modify(Yield* o) {
  if (o->resume) {
    o->resume = o->resume->accept(this);
  }

  /* construct the yield state, this being the parameters and local variables
   * that must be preserved for execution to resume */
  for (auto iter1 = scopes.rbegin(); iter1 != scopes.rend(); ++iter1) {
    auto& params = (*iter1)->parameters;
    auto& locals = (*iter1)->localVariables;

    for (auto pair : params) {
      auto param = pair.second;
      o->state.push_back(new NamedExpression(param->name, param->loc));
    }
    for (auto pair : locals) {
      auto local = pair.second;
      o->state.push_back(new NamedExpression(local->name, local->loc));
    }
  }
  return ScopedModifier::modify(o);
}
