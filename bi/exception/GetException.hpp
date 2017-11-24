/**
 * @file
 */
#pragma once

#include "bi/exception/CompilerException.hpp"
#include "bi/expression/Get.hpp"

namespace bi {
/**
 * Invalid use of "!" get operator.
 *
 * @ingroup birch_exception
 */
struct GetException: public CompilerException {
  /**
   * Constructor.
   */
  GetException(const Get* o);
};
}
