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

void birch::File::accept(Visitor* visitor) const {
  visitor->visit(this);
}
