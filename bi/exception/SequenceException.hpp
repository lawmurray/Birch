/**
 * @file
 */
#pragma once

#include "bi/exception/CompilerException.hpp"
#include "bi/expression/Sequence.hpp"

namespace bi {
/**
 * No common type for elements in a sequence.
 *
 * @ingroup compiler_exception
 */
struct SequenceException: public CompilerException {
  /**
   * Constructor.
   */
  SequenceException(const Sequence* o);
};
}
