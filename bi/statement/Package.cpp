/**
 * @file
 */
#include "bi/statement/Package.hpp"

#include "bi/visitor/all.hpp"

bi::Package::Package(const std::list<File*>& files) :
    files(files) {
  //
}

bi::Package::~Package() {
  //
}

bi::Package* bi::Package::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Package* bi::Package::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Package::accept(Visitor* visitor) const {
  visitor->visit(this);
}
