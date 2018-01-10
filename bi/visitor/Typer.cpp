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
  rootScope->add(o);
  return o;
}
