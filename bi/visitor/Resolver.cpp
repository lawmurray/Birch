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
  for (auto iter = scopes.rbegin(); iter != scopes.rend() && !o->category;
      ++iter) {
    (*iter)->lookup(o);
  }
  if (!o->category) {
    throw UnresolvedException(o);
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
