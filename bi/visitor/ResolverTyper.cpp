/**
 * @file
 */
#include "bi/visitor/ResolverTyper.hpp"

bi::ResolverTyper::ResolverTyper(Scope* rootScope) : rootScope(rootScope) {
  //
}

bi::ResolverTyper::~ResolverTyper() {
  //
}

bi::Statement* bi::ResolverTyper::modify(Basic* o) {
  rootScope->add(o);
  return o;
}

bi::Statement* bi::ResolverTyper::modify(Explicit* o) {
  return o;
}

bi::Statement* bi::ResolverTyper::modify(Class* o) {
  if (o->base->isEmpty() && o->name->str() != "Object") {
    /* if the class derives from nothing else, then derive from Object,
     * unless this is itself the declaration of the Object class */
    o->base = new ClassType(new Name("Object"), new EmptyType(), o->loc);
  }
  rootScope->add(o);
  return o;
}
