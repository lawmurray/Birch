/**
 * @file
 */
#pragma once

#include "bi/exception/CompilerException.hpp"
#include "bi/statement/Yield.hpp"
#include "bi/type/Type.hpp"

namespace bi {
/**
 * Incorrect number of dimensions in slice.
 *
 * @ingroup compiler_exception
 */
struct SliceException: public CompilerException {
  /**
   * Constructor.
   *
   * @param o Slice expression.
   * @param typeSize Number of dimensions in type.
   * @param sliceSize Number of dimensions in slice.
   */
  SliceException(const Expression* o, const int typeSize,
      const int sliceSize);
};
}
