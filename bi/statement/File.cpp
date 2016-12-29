/**
 * @file
 */
#include "bi/statement/File.hpp"

#include "bi/visitor/all.hpp"

bi::File::File(const std::string& path, Statement* imports, Statement* root) :
    path(path), imports(imports), root(root), state(UNRESOLVED) {
  //
}

bi::File* bi::File::acceptClone(Cloner* visitor) const {
  return visitor->clone(this);
}

void bi::File::acceptModify(Modifier* visitor) {
  visitor->modify(this);
}

void bi::File::accept(Visitor* visitor) const {
  visitor->visit(this);
}
