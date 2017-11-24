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
 * @ingroup birch_exception
 */
struct ConstructorException: public CompilerException {
  /**
   * Constructor.
   *
   * @param o Declaration.
   * @param type Class.
   */
  ConstructorException(const Argumented* o, const Class* type = nullptr);
};
}
