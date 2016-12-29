/**
 * @file
 */
#pragma once

#include "bi/exception/CompilerException.hpp"
#include "bi/statement/File.hpp"

namespace bi {
/**
 * Cyclic import.
 *
 * @ingroup compiler_exception
 *
 * @see poset
 */
struct CyclicImportException: public CompilerException {
  /**
   * Constructor.
   */
  CyclicImportException(File* o);
};
}
