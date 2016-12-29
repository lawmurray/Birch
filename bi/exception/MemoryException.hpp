/**
 * @file
 */
#pragma once

#include "bi/exception/Exception.hpp"

namespace bi {
/**
 * Memory exception.
 *
 * @ingroup compiler_exception
 */
struct MemoryException: public Exception {
  /**
   * Constructor.
   *
   * @param status NetCDF status code.
   */
  MemoryException(const std::string& msg);
};
}
