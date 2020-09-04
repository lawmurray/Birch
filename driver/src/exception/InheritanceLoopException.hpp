/**
 * @file
 */
#pragma once

#include "src/exception/Exception.hpp"
#include "src/statement/Class.hpp"

namespace birch {
/**
 * Loop in class inheritance.
 *
 * @ingroup exception
 */
struct InheritanceLoopException: public Exception {
  /**
   * Constructor.
   */
  InheritanceLoopException(const Class* o);
};
}
