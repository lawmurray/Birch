/**
 * @file
 */
#include "bi/visitor/Typer.hpp"

bi::Typer::Typer() :
    file(nullptr) {
  //
}

bi::Typer::~Typer() {
  //
}

bi::File* bi::Typer::modify(File* o) {
  file = o;
  return Modifier::modify(o);
}

bi::Statement* bi::Typer::modify(Basic* o) {
  assert(file);
  file->scope->add(o);
  return o;
}

bi::Statement* bi::Typer::modify(Class* o) {
  assert(file);
  file->scope->add(o);
  return o;
}

bi::Statement* bi::Typer::modify(Alias* o) {
  assert(file);
  file->scope->add(o);
  return o;
}
