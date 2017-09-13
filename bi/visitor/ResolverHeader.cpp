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

bi::Statement* bi::ResolverHeader::modify(ConversionOperator* o) {
  scopes.push_back(o->scope);
  o->returnType = o->returnType->accept(this);
  scopes.pop_back();
  currentClass->addConversion(o->returnType);
  return o;
}

bi::Statement* bi::ResolverHeader::modify(Class* o) {
  scopes.push_back(o->scope);
  currentClass = o;
  o->parens = o->parens->accept(this);
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

bi::Statement* bi::ResolverHeader::modify(Alias* o) {
  o->base = o->base->accept(this);
  return o;
}
