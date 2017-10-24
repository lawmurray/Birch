/**
 * @file
 */
#include "bi/visitor/Typer.hpp"

bi::Typer::Typer() : currentFile(nullptr) {
  //
}

bi::Typer::~Typer() {
  //
}

bi::File* bi::Typer::modify(File* o) {
  currentFile = o;
  o->root = o->root->accept(this);
  currentFile = nullptr;
  return o;
}

bi::Statement* bi::Typer::modify(Basic* o) {
  currentFile->scope->add(o);
  return o;
}

bi::Statement* bi::Typer::modify(Class* o) {
  o->typeParams = o->typeParams->accept(this);
  currentFile->scope->add(o);
  return o;
}

bi::Statement* bi::Typer::modify(Alias* o) {
  currentFile->scope->add(o);
  return o;
}
