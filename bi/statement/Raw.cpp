/**
 * @file
 */
#include "bi/statement/Raw.hpp"

#include "bi/visitor/all.hpp"

#include "boost/algorithm/string/trim.hpp"

#include <typeinfo>

bi::Raw::Raw(shared_ptr<Name> name, const std::string& raw,
    shared_ptr<Location> loc) :
    Statement(loc), Named(name), raw(raw) {
  boost::algorithm::trim_left(this->raw);
}

bi::Raw::~Raw() {
  //
}

bi::Statement* bi::Raw::acceptClone(Cloner* visitor) const {
  return visitor->clone(this);
}

void bi::Raw::acceptModify(Modifier* visitor) {
  visitor->modify(this);
}

void bi::Raw::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::Raw::operator<=(Statement& o) {
  try {
    Raw& o1 = dynamic_cast<Raw&>(o);
    return raw.compare(o1.raw) == 0;
  } catch (std::bad_cast e) {
    //
  }
  return false;
}

bool bi::Raw::operator==(const Statement& o) const {
  try {
    const Raw& o1 = dynamic_cast<const Raw&>(o);
    return raw.compare(o1.raw) == 0;
  } catch (std::bad_cast e) {
    //
  }
  return false;
}
