/**
 * @file
 */
#pragma once

#include "bi/exception/CompilerException.hpp"
#include "bi/type/Type.hpp"

namespace bi {
/**
 * Type that is not read-only, but should be.
 *
 * @ingroup birch_exception
 */
struct ReadOnlyException: public CompilerException {
  /**
   * Constructor.
   *
   * @param o Type.
   */
  ReadOnlyException(const Type* o);
};
}
