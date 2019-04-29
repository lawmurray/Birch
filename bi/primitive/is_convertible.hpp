/**
 * @file
 */
#pragma once

#include "bi/type/Type.hpp"
#include "bi/common/Parameterised.hpp"
#include "bi/common/Argumented.hpp"
#include "bi/common/Unknown.hpp"

namespace bi {
/**
 * Comparison of two objects using their isConvertible() function.
 */
struct is_convertible {
  bool operator()(Type* o1, Type* o2) {
    return o1->isConvertible(*o2);
  }

  bool operator()(Unknown* o1, Unknown* o2) {
    return false;
  }

  bool operator()(Parameterised* o1, Parameterised* o2) {
    return o1->params->type->isConvertible(*o2->params->type);
  }

  bool operator()(Argumented* o1, Parameterised* o2) {
    return o1->args->type->isConvertible(*o2->params->type);
  }
};
}
