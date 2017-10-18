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
  ///@todo Check that base type is of class type
  if (!o->base->isEmpty()) {
    o->addSuper(o->base);
  }
  o->braces = o->braces->accept(this);
  currentClass = nullptr;
  scopes.pop_back();
  return o;
}

bi::Statement* bi::ResolverSuper::modify(Alias* o) {
  o->base = o->base->accept(this);
  return o;
}

bi::Statement* bi::ResolverSuper::modify(Basic* o) {
  o->base = o->base->accept(this);
  ///@todo Check that base type is of basic type
  if (!o->base->isEmpty()) {
    o->addSuper(o->base);
  }
  return o;
}
