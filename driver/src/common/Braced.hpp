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
   * Destructor.
   */
  virtual ~Braced() = 0;

  /**
   * Body.
   */
  Statement* braces;
};
}
