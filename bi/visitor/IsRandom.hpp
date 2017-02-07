/**
 * @file
 */
#pragma once

#include "bi/visitor/Visitor.hpp"

namespace bi {
/**
 * Is this a rich expression?
 *
 * @ingroup compiler_visitor
 */
class IsRandom : public Visitor {
public:
  /**
   * Constructor.
   */
  IsRandom();

  /**
   * Destructor.
   */
  virtual ~IsRandom();

  virtual void visit(const RandomType* o);

  /**
   * Result.
   */
  bool result;
};
}
