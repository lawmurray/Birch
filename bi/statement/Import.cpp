/**
 * @file
 */
#include "bi/statement/Import.hpp"

#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::Import::Import(shared_ptr<Path> path, File* file,
    shared_ptr<Location> loc) :
    Statement(loc),
    path(path),
    file(file) {
  //
}

bi::Import::~Import() {
  //
}

bi::Statement* bi::Import::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Statement* bi::Import::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Import::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::Import::dispatch(Statement& o) {
  return o.le(*this);
}

bool bi::Import::le(Import& o) {
  return *path == *o.path;
}
