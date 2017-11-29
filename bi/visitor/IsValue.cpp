/**
 * @file
 */
#include "bi/visitor/IsValue.hpp"

bi::IsValue::~IsValue() {
  //
}

bool bi::IsValue::apply(const Type* o) {
  result = true;
  o->accept(this);
  return result;
}

void bi::IsValue::visit(const ClassType* o) {
  result = false;
}
