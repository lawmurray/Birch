/**
 * @file
 */
#pragma once

#include "src/exception/Exception.hpp"

namespace birch {
/**
 * File not found when parsing.
 *
 * @ingroup exception
 */
struct FileNotFoundException: public Exception {
  /**
   * Constructor.
   */
  FileNotFoundException(const std::string& name);
};
}
