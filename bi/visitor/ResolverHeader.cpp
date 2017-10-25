/**
 * @file
 */
#include "ResolverHeader.hpp"

bi::ResolverHeader::ResolverHeader() {
  //
}

bi::ResolverHeader::~ResolverHeader() {
  //
}

bi::Expression* bi::ResolverHeader::modify(Parameter* o) {
  Modifier::modify(o);
  scopes.back()->add(o);
  return o;
}

bi::Expression* bi::ResolverHeader::modify(MemberParameter* o) {
  Modifier::modify(o);
  scopes.back()->add(o);
  return o;
}

bi::Statement* bi::ResolverHeader::modify(GlobalVariable* o) {
  Modifier::modify(o);
  o->type->accept(&assigner);
  scopes.back()->add(o);
  return o;
}

bi::Statement* bi::ResolverHeader::modify(MemberVariable* o) {
  Modifier::modify(o);
  o->type->accept(&assigner);
  scopes.back()->add(o);
  return o;
}

bi::Statement* bi::ResolverHeader::modify(Function* o) {
  scopes.push_back(o->scope);
  o->params = o->params->accept(this);
  o->returnType = o->returnType->accept(this);
  o->type = new FunctionType(o->params->type->accept(&cloner),
      o->returnType->accept(&cloner), o->loc);
  o->type = o->type->accept(this);
  scopes.pop_back();
  scopes.back()->add(o);
  return o;
}

bi::Statement* bi::ResolverHeader::modify(Fiber* o) {
  scopes.push_back(o->scope);
  o->params = o->params->accept(this);
  o->returnType = o->returnType->accept(this);
  o->type = new FunctionType(o->params->type->accept(&cloner),
      o->returnType->accept(&cloner), o->loc);
  o->type = o->type->accept(this);
  scopes.pop_back();
  scopes.back()->add(o);
  if (!o->returnType->isFiber()) {
    throw FiberTypeException(o);
  }
  return o;
}

bi::Statement* bi::ResolverHeader::modify(Program* o) {
  scopes.push_back(o->scope);
  o->params = o->params->accept(this);
  scopes.pop_back();
  scopes.back()->add(o);
  ///@todo Check that can assign String to all option types
  return o;
}

bi::Statement* bi::ResolverHeader::modify(MemberFunction* o) {
  scopes.push_back(o->scope);
  o->params = o->params->accept(this);
  o->returnType = o->returnType->accept(this);
  o->type = new FunctionType(o->params->type->accept(&cloner),
      o->returnType->accept(&cloner), o->loc);
  o->type = o->type->accept(this);
  scopes.pop_back();
  scopes.back()->add(o);
  return o;
}

bi::Statement* bi::ResolverHeader::modify(MemberFiber* o) {
  scopes.push_back(o->scope);
  o->params = o->params->accept(this);
  o->returnType = o->returnType->accept(this);
  o->type = new FunctionType(o->params->type->accept(&cloner),
      o->returnType->accept(&cloner), o->loc);
  o->type = o->type->accept(this);
  scopes.pop_back();
  scopes.back()->add(o);
  if (!o->returnType->isFiber()) {
    throw FiberTypeException(o);
  }
  return o;
}

bi::Statement* bi::ResolverHeader::modify(BinaryOperator* o) {
  ///@todo Check that operator is in fact a binary operator, as parser no
  ///      longer distinguishes
  scopes.push_back(o->scope);
  o->params = o->params->accept(this);
  o->returnType = o->returnType->accept(this);
  o->type = new FunctionType(o->params->type->accept(&cloner),
      o->returnType->accept(&cloner), o->loc);
  o->type = o->type->accept(this);
  scopes.pop_back();
  scopes.back()->add(o);
  return o;
}

bi::Statement* bi::ResolverHeader::modify(UnaryOperator* o) {
  ///@todo Check that operator is in fact a unary operator, as parser no
  ///      longer distinguishes
  scopes.push_back(o->scope);
  o->params = o->params->accept(this);
  o->returnType = o->returnType->accept(this);
  o->type = new FunctionType(o->params->type->accept(&cloner),
      o->returnType->accept(&cloner), o->loc);
  o->type = o->type->accept(this);
  scopes.pop_back();
  scopes.back()->add(o);
  return o;
}

bi::Statement* bi::ResolverHeader::modify(AssignmentOperator* o) {
  scopes.push_back(o->scope);
  o->single = o->single->accept(this);
  scopes.pop_back();
  currentClass->addAssignment(o->single->type);
  return o;
}

bi::Statement* bi::ResolverHeader::modify(Class* o) {
  scopes.push_back(o->scope);
  currentClass = o;
  o->params = o->params->accept(this);
  if (o->typeParams->isEmpty()) {
    /* ^ otherwise uses generics, braces will be handled on instantiation */
    o->braces = o->braces->accept(this);
  }
  currentClass = nullptr;
  scopes.pop_back();
  return o;
}
