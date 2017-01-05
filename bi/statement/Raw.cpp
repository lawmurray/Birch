/**
 * @file
 */
#include "bi/statement/Raw.hpp"

#include "bi/visitor/all.hpp"

#include "boost/algorithm/string/trim.hpp"

#include <typeinfo>

bi::Raw::Raw(shared_ptr<Name> name, const std::string& raw,
    shared_ptr<Location> loc) :
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

bool bi::Raw::dispatch(Statement& o) {
  return o.le(*this);
}

bool bi::Raw::le(Raw& o) {
  return raw.compare(o.raw) == 0;
}
