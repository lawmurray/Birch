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
   * @param msg Message.
   */
  MemoryException(const std::string& msg);
};
}
