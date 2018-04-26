/**
 * @file
 */
#pragma once

#include "bi/exception/CompilerException.hpp"
#include "bi/statement/ConversionOperator.hpp"

namespace bi {
/**
 * Invalid type for conversion operator declaration.
 *
 * @ingroup exception
 */
struct ConversionOperatorException: public CompilerException {
  /**
   * Constructor.
   */
  ConversionOperatorException(const ConversionOperator* o);
};
}
