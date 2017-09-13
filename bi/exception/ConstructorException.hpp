/**
 * @file
 */
#pragma once

#include "bi/exception/CompilerException.hpp"
#include "bi/type/Type.hpp"
#include "bi/statement/Class.hpp"

namespace bi {
/**
 * Invalid constructor call.
 *
 * @ingroup compiler_exception
 */
struct ConstructorException: public CompilerException {
  /**
   * Constructor.
   *
   * @param args Arguments.
   * @param type Class.
   */
  ConstructorException(const Type* args, const Class* type = nullptr);
};
}
