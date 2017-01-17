/**
 * @file
 */
#pragma once

#include "bi/primitive/possibly.hpp"

namespace bi {
/**
 * Parameter.
 *
 * @tparam Argument Argument type.
 *
 * @ingroup compiler_common
 */
template<class Argument>
class Parameter {
public:
  /**
   * Constructor.
   */
  Parameter();

  /**
   * Destructor.
   */
  virtual ~Parameter() = 0;

  /**
   * Capture an argument.
   *
   * @param arg The argument.
   *
   * @return True.
   */
  possibly capture(Argument* arg);

  /**
   * Argument.
   */
  Argument* arg;
};
}
