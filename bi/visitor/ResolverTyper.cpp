/**
 * @file
 */
#include "bi/visitor/ResolverTyper.hpp"

bi::ResolverTyper::ResolverTyper(Scope* rootScope) {
  scopes.push_back(rootScope);
}

bi::ResolverTyper::~ResolverTyper() {
  scopes.pop_back();
}

bi::Expression* bi::ResolverTyper::modify(Generic* o) {
  Modifier::modify(o);
  scopes.back()->add(o);
  return o;
}

bi::Statement* bi::ResolverTyper::modify(Basic* o) {
  Modifier::modify(o);
  scopes.back()->add(o);
  return o;
}

bi::Statement* bi::ResolverTyper::modify(Explicit* o) {
  return o;
}

bi::Statement* bi::ResolverTyper::modify(Class* o) {
  scopes.push_back(o->scope);
  o->typeParams = o->typeParams->accept(this);
  if (o->base->isEmpty() && o->name->str() != "Object") {
    /* if the class derives from nothing else, then derive from Object,
     * unless this is itself the declaration of the Object class */
    o->base = new ClassType(new Name("Object"), new EmptyType(), o->loc);
  }
  o->state = RESOLVED_TYPER;
  scopes.pop_back();
  if (!o->isInstantiation()) {
    scopes.back()->add(o);
  }
  return o;
}
