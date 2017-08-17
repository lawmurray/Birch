/**
 * @file
 */
#include "bi/visitor/Importer.hpp"

bool bi::Importer::import(File* file) {
  this->file = file;
  this->haveNew = false;
  this->file->accept(this);
  return this->haveNew;
}

bi::Importer::~Importer() {
  //
}

bi::Statement* bi::Importer::modify(Import* o) {
  haveNew = file->scope->import(o->file->scope) || haveNew;
  return o;
}
