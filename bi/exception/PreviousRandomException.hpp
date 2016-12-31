/**
 * @file
 */
#pragma once

#include "bi/exception/CompilerException.hpp"
#include "bi/expression/RandomVariable.hpp"

namespace bi {
/**
 * Previous random specification for the same variable.
 *
 * @ingroup compiler_exception
 */
struct PreviousRandomException: public CompilerException {
  /**
   * Constructor.
   */
  PreviousRandomException(RandomVariable* random, RandomVariable* prev);
};
}
