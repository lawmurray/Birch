/**
 * @file
 */
#include "bi/visitor/Typer.hpp"

bi::Typer::Typer(Scope* rootScope) : rootScope(rootScope) {
  //
}

bi::Typer::~Typer() {
  //
}

bi::Statement* bi::Typer::modify(Basic* o) {
  rootScope->add(o);
  return o;
}

bi::Statement* bi::Typer::modify(Class* o) {
  if (o->base->isEmpty() && o->name->str() != "Object") {
    /* if the class derives from nothing else, then derive from Object,
     * unless this is itself the declaration of the Object class */
    o->base = new ClassType(new Name("Object"), new EmptyType(), o->loc);
  }
  rootScope->add(o);
  return o;
}
