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

void birch::Raw::accept(Visitor* visitor) const {
  visitor->visit(this);
}
