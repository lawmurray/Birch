/**
 * @file
 */
#include "ResolverSuper.hpp"

bi::ResolverSuper::ResolverSuper() {
  //
}

bi::ResolverSuper::~ResolverSuper() {
  //
}

bi::Statement* bi::ResolverSuper::modify(ConversionOperator* o) {
  o->returnType = o->returnType->accept(this);
  currentClass->addConversion(o->returnType);
  return o;
}

bi::Statement* bi::ResolverSuper::modify(Class* o) {
  scopes.push_back(o->scope);
  currentClass = o;
  o->base = o->base->accept(this);
  if (!o->base->isEmpty()) {
    o->addSuper(o->base);
  }
  o->braces = o->braces->accept(this);
  currentClass = nullptr;
  scopes.pop_back();
  ///@todo Check that base type is of class type
  return o;
}

bi::Statement* bi::ResolverSuper::modify(Alias* o) {
  o->base = o->base->accept(this);
  return o;
}
