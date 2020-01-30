/**
 * @file
 */
#include "bi/visitor/IsValue.hpp"

bi::IsValue::IsValue() : result(true) {
  //
}

void bi::IsValue::visit(const NamedType* o) {
  ///@todo
  result = false;
}

void bi::IsValue::visit(const FiberType* o) {
  result = false;  // state is an object
}

void bi::IsValue::visit(const Call* o) {
  ///@todo
  result = false;
}
