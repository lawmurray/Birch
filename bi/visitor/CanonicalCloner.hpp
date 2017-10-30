/**
 * @file
 */
#pragma once

#include "bi/visitor/Cloner.hpp"

namespace bi {
/**
 * Cloner that replaces generic types with their arguments.
 *
 * @ingroup compiler_visitor
 */
class CanonicalCloner : public Cloner {
public:
  virtual Type* clone(const GenericType* o);
};
}
