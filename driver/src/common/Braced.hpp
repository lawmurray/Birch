/**
 * @file
 */
#pragma once

#include "src/statement/Statement.hpp"

namespace birch {
/**
 * Object with braces.
 *
 * @ingroup common
 */
class Braced {
public:
  /**
   * Constructor.
   *
   * @param braces Body.
   */
  Braced(Statement* braces);

  /**
   * Body.
   */
  Statement* braces;
};
}
