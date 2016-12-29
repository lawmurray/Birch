/**
 * @file
 */
#pragma once

#include "bi/exception/Exception.hpp"

namespace bi {
/**
 * Unknown option given to program.
 *
 * @ingroup compiler_exception
 */
struct UnknownOptionException: public Exception {
  /**
   * Constructor.
   *
   * @param status NetCDF status code.
   */
  UnknownOptionException(const std::string& option);
};
}
