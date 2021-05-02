/**
 * @file
 */
#include "src/statement/File.hpp"

#include "src/visitor/all.hpp"

birch::File::File(const std::string& path, Statement* root) :
    path(path),
    root(root) {
  //
}

birch::File* birch::File::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void birch::File::accept(Visitor* visitor) const {
  visitor->visit(this);
}
