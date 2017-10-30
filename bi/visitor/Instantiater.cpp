/**
 * @file
 */
#include "bi/visitor/CanonicalCloner.hpp"

bi::Type* bi::CanonicalCloner::clone(const GenericType* o) {
  return o->target->type->accept(this);
}
