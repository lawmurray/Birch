/**
 * @file
 */
#include "bi/statement/File.hpp"

#include "bi/visitor/all.hpp"

bi::File::File(const std::string& path, Scope* scope, Statement* root) :
    Scoped(scope),
    path(path),
    root(root) {
  assert(scope->category == GLOBAL_SCOPE);
}

bi::File::~File() {
  //
}

bi::File* bi::File::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::File* bi::File::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::File::accept(Visitor* visitor) const {
  visitor->visit(this);
}
