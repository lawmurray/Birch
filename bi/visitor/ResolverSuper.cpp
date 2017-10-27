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

bi::Expression* bi::ResolverSuper::modify(Generic* o) {
  o->type = o->type->accept(this);
  scopes.back()->add(o);
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
bi::Statement* bi::ResolverSuper::modify(Class* o) {
  scopes.push_back(o->scope);
  currentClass = o;
  o->typeParams = o->typeParams->accept(this);
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
  return o;
}

bi::Statement* bi::ResolverSuper::modify(GlobalVariable* o) {
  return o;
}

bi::Statement* bi::ResolverSuper::modify(Function* o) {
  return o;
}

bi::Statement* bi::ResolverSuper::modify(Fiber* o) {
  return o;
}

bi::Statement* bi::ResolverSuper::modify(Program* o) {
  return o;
}

bi::Statement* bi::ResolverSuper::modify(BinaryOperator* o) {
  return o;
}

bi::Statement* bi::ResolverSuper::modify(UnaryOperator* o) {
  return o;
}

bi::Statement* bi::ResolverSuper::modify(MemberVariable* o) {
  return o;
}

bi::Statement* bi::ResolverSuper::modify(MemberFunction* o) {
  return o;
}

bi::Statement* bi::ResolverSuper::modify(MemberFiber* o) {
  return o;
}

bi::Statement* bi::ResolverSuper::modify(AssignmentOperator* o) {
  return o;
}

bi::Statement* bi::ResolverSuper::modify(ConversionOperator* o) {
  o->returnType = o->returnType->accept(this);
  currentClass->addConversion(o->returnType);
  return o;
}
