/**
 * @file
 */
#pragma once

#include "bi/exception/CompilerException.hpp"
#include "bi/type/ClassType.hpp"
#include "bi/statement/Class.hpp"

namespace bi {
/**
 * Invalid generic type arguments.
 *
 * @ingroup exception
 */
struct GenericException: public CompilerException {
  /**
   * Constructor.
   *
   * @param ref Class type.
   * @param param Class.
   */
  GenericException(const ClassType* ref, const Class* param);
};
}
