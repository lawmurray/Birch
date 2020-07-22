/**
 * @file
 */
#include "bi/statement/Raw.hpp"

#include "bi/visitor/all.hpp"

bi::Raw::Raw(Name* name, const std::string& raw,
    Location* loc) :
    Statement(loc),
    Named(name),
    raw(raw) {
  boost::algorithm::trim_left(this->raw);
}

bi::Raw::~Raw() {
  //
}

bi::Statement* bi::Raw::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Statement* bi::Raw::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Raw::accept(Visitor* visitor) const {
  visitor->visit(this);
}
