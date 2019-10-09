/**
 * @file
 */
#pragma once

#include "bi/visitor/Visitor.hpp"

namespace bi {
/**
 * Determines whether an expression, statement or type contains contains
 * any class types.
 *
 * @ingroup visitor
 */
class IsValue: public Visitor {
public:
  using Visitor::visit;

  IsValue();

  virtual void visit(const ClassType* o);
  virtual void visit(const GenericType* o);

  /**
   * Result.
   */
  bool result;
};
}
