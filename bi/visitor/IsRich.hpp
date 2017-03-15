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
class IsRich : public Visitor {
public:
  /**
   * Constructor.
   */
  IsRich();

  /**
   * Destructor.
   */
  virtual ~IsRich();

  using Visitor::visit;
  virtual void visit(const BracesExpression* o);
  virtual void visit(const BracketsExpression* o);
  virtual void visit(const Index* o);
  virtual void visit(const Range* o);
  virtual void visit(const Member* o);
  virtual void visit(const This* o);
  virtual void visit(const LambdaInit* o);
  virtual void visit(const RandomInit* o);
  virtual void visit(const VarReference* o);
  virtual void visit(const FuncReference* o);

  /**
   * Result.
   */
  bool result;
};
}
