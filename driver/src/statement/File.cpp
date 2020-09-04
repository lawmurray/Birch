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

birch::File::~File() {
  //
}

birch::File* birch::File::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

birch::File* birch::File::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void birch::File::accept(Visitor* visitor) const {
  visitor->visit(this);
}
