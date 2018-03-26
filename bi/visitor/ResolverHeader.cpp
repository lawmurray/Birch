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

bi::Statement* bi::ResolverHeader::modify(Explicit* o) {
  o->base = o->base->accept(this);
  assert(o->base->getClass());
  o->base->getClass()->isExplicit = true;
  return o;
}

bi::Statement* bi::ResolverHeader::modify(Class* o) {
  if (o->state < RESOLVED_SUPER) {
    ResolverSuper resolver(scopes.front());
    o->accept(&resolver);
  }
  if (o->state < RESOLVED_HEADER) {
      /* actual type check skipped for inheritance in ResolverSuper, so do
       * that now */
    auto base = dynamic_cast<ClassType*>(o->base);
    if (base && base->original &&
        !base->typeArgs->definitely(*base->original->typeParams->type)) {
      throw GenericException(base, base->original);
    }

    if (!o->base->isEmpty()) {
      o->scope->inherit(o->base->getClass()->scope);
    }
    classes.push_back(o);
    scopes.push_back(o->scope);
    scopes.push_back(o->initScope);
    if (o->isAlias()) {
      o->params =
          o->base->canonical()->getClass()->params->accept(&cloner)->accept(
              this);
    } else {
      o->params = o->params->accept(this);
    }
    scopes.pop_back();
    if (o->isBound()) {
      o->braces = o->braces->accept(this);
    }
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
  o->returnType = new FiberType(o->returnType->accept(this));
  o->type = new FunctionType(o->params->type, o->returnType, o->loc);
  scopes.pop_back();
  scopes.back()->add(o);
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
  o->returnType = new FiberType(o->returnType->accept(this));
  o->type = new FunctionType(o->params->type, o->returnType, o->loc);
  scopes.pop_back();
  scopes.back()->add(o);
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

