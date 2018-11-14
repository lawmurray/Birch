/**
 * @file
 */
#pragma once

#include "bi/common/Annotated.hpp"
#include "bi/common/Parameterised.hpp"
#include "bi/common/ReturnTyped.hpp"

namespace bi {
/**
 * Placeholder for unknown target type.
 */
class Unknown: public Annotated, public Parameterised, public ReturnTyped {
  //
};
}
