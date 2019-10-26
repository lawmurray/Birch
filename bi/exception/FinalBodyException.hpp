/**
 * @file
 */
#pragma once

#include "bi/exception/CompilerException.hpp"
#include "bi/statement/MemberFunction.hpp"

namespace bi {
/**
 * Member function declared final, but without a body.
 *
 * @ingroup exception
 */
struct FinalBodyException: public CompilerException {
  /**
   * Constructor.
   */
  FinalBodyException(const MemberFunction* o);
};
}
