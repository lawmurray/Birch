/**
 * @file
 */
#pragma once

#include "bi/exception/Exception.hpp"

namespace bi {
/**
 * Driver exception.
 *
 * @ingroup compiler_exception
 */
struct DriverException: public Exception {
  /**
   * Default constructor.
   */
  DriverException();

  /**
   * Constructor.
   *
   * @param msg Message.
   */
  DriverException(const std::string& msg);
};
}
