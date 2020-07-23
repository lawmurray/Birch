/**
 * @file
 */
#pragma once

#include "bi/exception/Exception.hpp"
#include "bi/statement/Class.hpp"

namespace bi {
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
