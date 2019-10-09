/**
 * @file
 */
#include "bi/visitor/IsValue.hpp"

bi::IsValue::IsValue() : result(true) {

}

void bi::IsValue::visit(const ClassType* o) {
  result = false;
}

void bi::IsValue::visit(const GenericType* o) {
  o->target->type->accept(this);
}
