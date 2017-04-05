/**
 * @file
 */
#pragma once

#include "bi/visitor/Visitor.hpp"

namespace bi {
/**
 * Determine if a model is of polymorphic type.
 *
 * @ingroup compiler_visitor
 */
class IsPolymorphic : public Visitor {
public:
  /**
   * Constructor.
   */
  IsPolymorphic();

  /**
   * Destructor.
   */
  virtual ~IsPolymorphic();

  using Visitor::visit;

  virtual void visit(const FuncDeclaration* o);

  /**
   * The result, after visiting.
   */
  bool result;
};
}
