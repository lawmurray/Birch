/**
 * @file
 */
#include "bi/statement/Import.hpp"

#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::Import::Import(shared_ptr<Path> path, File* file, shared_ptr<Location> loc) :
    Statement(loc), path(path), file(file) {
  //
}

bi::Statement* bi::Import::acceptClone(Cloner* visitor) const {
  return visitor->clone(this);
}

void bi::Import::acceptModify(Modifier* visitor) {
  visitor->modify(this);
}

void bi::Import::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::Import::operator<=(Statement& o) {
  try {
    Import& o1 = dynamic_cast<Import&>(o);
    return *path == *o1.path;
  } catch (std::bad_cast e) {
    //
  }
  return false;
}

bool bi::Import::operator==(const Statement& o) const {
  try {
    const Import& o1 = dynamic_cast<const Import&>(o);
    return *path == *o1.path;
  } catch (std::bad_cast e) {
    //
  }
  return false;
}
