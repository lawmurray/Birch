/**
 * @file
 */
#pragma once

#include "bi/exception/Exception.hpp"

namespace bi {
/**
 * Compiler exception.
 *
 * @ingroup birch_exception
 */
struct CompilerException: public Exception {
  /**
   * Default constructor.
   */
  CompilerException();

  /**
   * Constructor.
   *
   * @param msg Message.
   */
  CompilerException(const std::string& msg);
};
}
