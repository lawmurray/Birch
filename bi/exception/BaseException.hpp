/**
 * @file
 */
#pragma once

#include "bi/exception/CompilerException.hpp"
#include "bi/statement/Basic.hpp"
#include "bi/statement/Class.hpp"

namespace bi {
/**
 * Invalid super type.
 *
 * @ingroup exception
 */
struct BaseException: public CompilerException {
  /**
   * Constructor.
   */
  BaseException(const Basic* o);

  /**
   * Constructor.
   */
  BaseException(const Class* o);
};
}
