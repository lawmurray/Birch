/**
 * @file
 */
#include "bi/visitor/IsReadOnly.hpp"

bi::IsReadOnly::IsReadOnly() : result(true) {
  //
}

void bi::IsReadOnly::visit(const PointerType* o) {
  Visitor::visit(o);
  result = result && o->read;
}
