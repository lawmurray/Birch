/**
 * @file
 */
#include "bi/visitor/Typer.hpp"

bi::Typer::Typer() {
  //
}

bi::Typer::~Typer() {
  //
}

bi::File* bi::Typer::modify(File* o) {
  files.push(o);
  o->root = o->root->accept(this);
  files.pop();
  return o;
}

bi::Statement* bi::Typer::modify(Basic* o) {
  files.top()->scope->add(o);
  return o;
}

bi::Statement* bi::Typer::modify(Class* o) {
  files.top()->scope->add(o);
  return o;
}

bi::Statement* bi::Typer::modify(Alias* o) {
  files.top()->scope->add(o);
  return o;
}
