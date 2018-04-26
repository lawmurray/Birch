/**
 * @file
 */
#include "bi/visitor/ResolverSuper.hpp"

#include "bi/visitor/Instantiater.hpp"

bi::ResolverSuper::ResolverSuper(Scope* rootScope) :
    Resolver(rootScope, false) {
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
  if (!o->base->isEmpty()) {
    if (!o->base->isBasic()) {
      throw BaseException(o);
    }
    o->addSuper(o->base);
  }
  return o;
}

bi::Statement* bi::ResolverSuper::modify(Explicit* o) {
  return o;
}

bi::Statement* bi::ResolverSuper::modify(Class* o) {
  if (o->state < RESOLVED_SUPER) {
    scopes.push_back(o->scope);
    classes.push_back(o);
    o->typeParams = o->typeParams->accept(this);
    o->base = o->base->accept(this);
    if (!o->base->isEmpty()) {
      if (!o->base->isClass()) {
        throw BaseException(o);
      }
      o->addSuper(o->base);
    }
    o->braces = o->braces->accept(this);
    o->state = RESOLVED_SUPER;
    classes.pop_back();
    scopes.pop_back();
  }
  for (auto instantiation : o->instantiations) {
    instantiation->accept(this);
  }
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
  if (!o->returnType->isValue()) {
    throw ConversionOperatorException(o);
  }
  classes.back()->addConversion(o->returnType);
  return o;
}

void bi::ResolverSuper::instantiate(ClassType* o) {
  if (!o->typeArgs->isEmpty() || o->target->isGeneric()) {
    // the next check differs from Resolver::instantiate(), it is a simple
    // check on the number of arguments, but not their type, as super
    // type relationships are still being resolved
    if (o->typeArgs->width() == o->target->typeParams->width()) {
      Class* instantiation = o->target->getInstantiation(o->typeArgs);
      if (!instantiation) {
        Instantiater instantiater(o->typeArgs);
        instantiation =
            dynamic_cast<Class*>(o->target->accept(&instantiater));
        assert(instantiation);
        instantiation->isInstantiation = true;
        o->target->addInstantiation(instantiation);
        instantiation->accept(this);
      }
      o->original = o->target;
      o->target = instantiation;
    } else {
      throw GenericException(o, o->target);
    }
  }
}
