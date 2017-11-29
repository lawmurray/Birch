/**
 * @file
 */
#pragma once

#include "bi/visitor/Visitor.hpp"

namespace bi {
/**
 * Determines whether a type contains only value types.
 *
 * @ingroup birch_visitor
 */
class IsValue: public Visitor {
public:
  /**
   * Destructor.
   */
  virtual ~IsValue();

  /**
   * Apply the visitor.
   */
  bool apply(const Type* o);

  using Visitor::visit;

  virtual void visit(const ClassType* o);

protected:
  /**
   * The result.
   */
  bool result;
};
}
