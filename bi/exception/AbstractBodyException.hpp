/**
 * @file
 */
#pragma once

#include "bi/exception/CompilerException.hpp"
#include "bi/statement/MemberFunction.hpp"

namespace bi {
/**
 * Member function declared abstract, but with a body.
 *
 * @ingroup exception
 */
struct AbstractBodyException: public CompilerException {
  /**
   * Constructor.
   */
  AbstractBodyException(const MemberFunction* o);
};
}
