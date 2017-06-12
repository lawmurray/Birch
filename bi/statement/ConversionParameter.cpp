/**
 * @file
 */
#include "bi/statement/ConversionParameter.hpp"

#include "bi/visitor/all.hpp"

bi::ConversionParameter::ConversionParameter(Type* type, Expression* braces,
    shared_ptr<Location> loc) :
    Statement(loc),
    Typed(type),
    Braced(braces) {
  //
}

bi::ConversionParameter::~ConversionParameter() {
  //
}

bi::Statement* bi::ConversionParameter::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Statement* bi::ConversionParameter::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::ConversionParameter::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::ConversionParameter::dispatchDefinitely(const Statement& o) const {
  return o.definitely(*this);
}

bool bi::ConversionParameter::definitely(const ConversionParameter& o) const {
  return type->definitely(*o.type);
}

bool bi::ConversionParameter::dispatchPossibly(const Statement& o) const {
  return o.possibly(*this);
}

bool bi::ConversionParameter::possibly(const ConversionParameter& o) const {
  return type->possibly(*o.type);
}
