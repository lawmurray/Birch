/**
 * @file
 */
#pragma once

#include "src/exception/Exception.hpp"

namespace birch {
/**
 * Driver exception.
 *
 * @ingroup exception
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
