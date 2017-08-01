/**
 * @file
 */
#include "bi/statement/Import.hpp"

#include "bi/visitor/all.hpp"

bi::Import::Import(Path* path, File* file,
    Location* loc) :
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
