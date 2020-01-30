/**
 * @file
 */
#pragma once

#include "bi/visitor/Visitor.hpp"

#include <unordered_set>

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

  virtual void visit(const NamedType* o);
  virtual void visit(const FiberType* o);
  virtual void visit(const Call* o);

  /**
   * Result.
   */
  bool result;

private:
  /**
   * Memo of functions into which have already recursed.
   */
  std::unordered_set<Statement*> done;
};
}
