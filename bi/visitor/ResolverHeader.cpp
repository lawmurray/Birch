/**
 * @file
 */
#include "bi/visitor/ResolverHeader.hpp"

#include "bi/visitor/ResolverSuper.hpp"

bi::ResolverHeader::ResolverHeader(Scope* rootScope) :
    Resolver(rootScope, true) {
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

bi::Expression* bi::ResolverHeader::modify(Generic* o) {
  return o;
}

bi::Statement* bi::ResolverHeader::modify(Basic* o) {
  return o;
}

bi::Statement* bi::ResolverHeader::modify(Class* o) {
  if (o->state < RESOLVED_SUPER) {
    ResolverSuper resolver(scopes.front());
    o->accept(&resolver);
  }
  if (o->state < RESOLVED_HEADER) {
    if (!o->base->isEmpty()) {
      o->scope->inherit(o->base->getClass()->scope);
    }
    scopes.push_back(o->scope);
    classes.push_back(o);
    if (o->isAlias()) {
      o->params = o->base->canonical()->getClass()->params->accept(&cloner)->accept(this);
    } else {
      o->params = o->params->accept(this);
    }
    o->braces = o->braces->accept(this);
    o->state = RESOLVED_HEADER;
    classes.pop_back();
    scopes.pop_back();
  }
  for (auto instantiation : o->instantiations) {
    instantiation->accept(this);
  }
  return o;
}

bi::Statement* bi::ResolverHeader::modify(GlobalVariable* o) {
  o->type = o->type->accept(this);
  if (!o->type->isReadOnly()) {
    throw ReadOnlyException(o->type);
  }
  if (!o->brackets->isEmpty()) {
    o->type = new ArrayType(o->type, o->brackets->width(), o->brackets->loc);
  }
  scopes.back()->add(o);
  return o;
}

bi::Statement* bi::ResolverHeader::modify(Function* o) {
  scopes.push_back(o->scope);
  o->params = o->params->accept(this);
  o->returnType = o->returnType->accept(this);
  o->type = new FunctionType(o->params->type, o->returnType, o->loc);
  scopes.pop_back();
  scopes.back()->add(o);
  return o;
}

bi::Statement* bi::ResolverHeader::modify(Fiber* o) {
  scopes.push_back(o->scope);
  o->params = o->params->accept(this);
  if (o->isClosed() && !o->params->type->isReadOnly()) {
    throw ReadOnlyException(o->params->type);
  }
  o->returnType = o->returnType->accept(this);
  if (o->isClosed() && !o->returnType->isReadOnly()) {
    throw ReadOnlyException(o->returnType);
  }
  o->type = new FunctionType(o->params->type, o->returnType, o->loc);
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

bi::Statement* bi::ResolverHeader::modify(BinaryOperator* o) {
  ///@todo Check that operator is in fact a binary operator, as parser no
  ///      longer distinguishes
  scopes.push_back(o->scope);
  o->params = o->params->accept(this);
  o->returnType = o->returnType->accept(this);
  o->type = new FunctionType(o->params->type, o->returnType, o->loc);
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
  o->type = new FunctionType(o->params->type, o->returnType, o->loc);
  scopes.pop_back();
  scopes.back()->add(o);
  return o;
}

bi::Expression* bi::ResolverHeader::modify(MemberParameter* o) {
  Modifier::modify(o);
  scopes.back()->add(o);
  return o;
}

bi::Statement* bi::ResolverHeader::modify(MemberVariable* o) {
  o->type = o->type->accept(this);
  if (!o->brackets->isEmpty()) {
    o->type = new ArrayType(o->type, o->brackets->width(), o->brackets->loc);
  }
  scopes.back()->add(o);
  return o;
}

bi::Statement* bi::ResolverHeader::modify(MemberFunction* o) {
  scopes.push_back(o->scope);
  o->params = o->params->accept(this);
  o->returnType = o->returnType->accept(this);
  o->type = new FunctionType(o->params->type, o->returnType, o->loc);
  scopes.pop_back();
  scopes.back()->add(o);
  return o;
}

bi::Statement* bi::ResolverHeader::modify(MemberFiber* o) {
  scopes.push_back(o->scope);
  o->params = o->params->accept(this);
  if (o->isClosed() && !o->params->type->isReadOnly()) {
    throw ReadOnlyException(o->params->type);
  }
  o->returnType = o->returnType->accept(this);
  if (o->isClosed() && !o->returnType->isReadOnly()) {
    throw ReadOnlyException(o->returnType);
  }
  o->type = new FunctionType(o->params->type, o->returnType, o->loc);
  scopes.pop_back();
  scopes.back()->add(o);
  if (!o->returnType->isFiber()) {
    throw FiberTypeException(o);
  }
  return o;
}

bi::Statement* bi::ResolverHeader::modify(AssignmentOperator* o) {
  scopes.push_back(o->scope);
  o->single = o->single->accept(this);
  scopes.pop_back();
  classes.back()->addAssignment(o->single->type);
  return o;
}

bi::Statement* bi::ResolverHeader::modify(ConversionOperator* o) {
  return o;
}

