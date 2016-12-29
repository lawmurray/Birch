/**
 * @file
 */
#pragma once

#include "bi/exception/Exception.hpp"

namespace bi {
/**
 * Compiler exception.
 *
 * @ingroup compiler_exception
 */
struct CompilerException: public Exception {
  /**
   * Default constructor.
   */
  CompilerException();

  /**
   * Constructor.
   *
   * @param status NetCDF status code.
   */
  CompilerException(const std::string& msg);
};
}
