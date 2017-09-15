/**
 * @file
 */
#pragma once

#include "bi/type/Type.hpp"
#include "bi/common/Parameterised.hpp"
#include "bi/common/Argumented.hpp"

namespace bi {
/**
 * Comparison of two objects using their definitely() function.
 */
struct definitely {
  bool operator()(Type* o1, Type* o2) {
    return o1->definitely(*o2);
  }

  bool operator()(Parameterised* o1, Parameterised* o2) {
    return o1->params->type->definitely(*o2->params->type);
  }

  bool operator()(Argumented* o1, Parameterised* o2) {
    return o1->args->type->definitely(*o2->params->type);
  }
};
}
