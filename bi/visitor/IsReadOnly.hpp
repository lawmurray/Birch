/**
 * @file
 */
#pragma once

#include "bi/visitor/Visitor.hpp"

namespace bi {
/**
 * Checks if a type is read-only.
 *
 * @ingroup birch_visitor
 */
class IsReadOnly: public Visitor {
public:
  /**
   * Constructor.
   */
  IsReadOnly();

  using Visitor::visit;
  virtual void visit(const PointerType* o);

  /**
   * Result.
   */
  bool result;
};
}
