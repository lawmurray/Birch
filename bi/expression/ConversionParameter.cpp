/**
 * @file
 */
#include "bi/expression/ConversionParameter.hpp"

#include "bi/visitor/all.hpp"

bi::ConversionParameter::ConversionParameter(Type* type, Expression* braces,
    shared_ptr<Location> loc) :
    Expression(type, loc),
    Braced(braces) {
  //
}

bi::ConversionParameter::~ConversionParameter() {
  //
}

bi::Expression* bi::ConversionParameter::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::ConversionParameter::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::ConversionParameter::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::ConversionParameter::dispatchDefinitely(const Expression& o) const {
  return o.definitely(*this);
}

bool bi::ConversionParameter::definitely(const ConversionParameter& o) const {
  return type->definitely(*o.type);
}

bool bi::ConversionParameter::dispatchPossibly(const Expression& o) const {
  return o.possibly(*this);
}

bool bi::ConversionParameter::possibly(const ConversionParameter& o) const {
  return type->possibly(*o.type);
}
