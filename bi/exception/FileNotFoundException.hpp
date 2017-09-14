/**
 * @file
 */
#pragma once

#include "bi/exception/Exception.hpp"

namespace bi {
/**
 * File not found when parsing.
 *
 * @ingroup compiler_exception
 */
struct FileNotFoundException: public Exception {
  /**
   * Constructor.
   */
  FileNotFoundException(const std::string& name);
};
}
