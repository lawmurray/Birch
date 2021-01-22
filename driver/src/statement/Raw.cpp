/**
 * @file
 */
#include "src/statement/Raw.hpp"

#include "src/visitor/all.hpp"

birch::Raw::Raw(Name* name, const std::string& raw,
    Location* loc) :
    Statement(loc),
    Named(name),
    raw(raw) {
  //
}

birch::Raw::~Raw() {
  //
}

birch::Statement* birch::Raw::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

birch::Statement* birch::Raw::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void birch::Raw::accept(Visitor* visitor) const {
  visitor->visit(this);
}
