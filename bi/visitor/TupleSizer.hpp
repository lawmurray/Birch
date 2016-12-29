/**
 * @file
 */
#pragma once

#include "bi/visitor/Visitor.hpp"

namespace bi {
/**
 * Visitor to count size of tuples.
 *
 * @ingroup compiler_visitor
 */
class TupleSizer : public Visitor {
public:
  /**
   * Constructor.
   */
  TupleSizer();

  /**
   * Destructor.
   */
  virtual ~TupleSizer();

  virtual void visit(const EmptyExpression* o);

  virtual void visit(const BoolLiteral* o);
  virtual void visit(const IntLiteral* o);
  virtual void visit(const RealLiteral* o);
  virtual void visit(const StringLiteral* o);
  virtual void visit(const ExpressionList* o);
  virtual void visit(const ParenthesesExpression* o);
  virtual void visit(const BracesExpression* o);
  virtual void visit(const Range* o);
  virtual void visit(const Traversal* o);
  virtual void visit(const This* o);
  virtual void visit(const BracketsExpression* o);

  virtual void visit(const VarReference* o);
  virtual void visit(const FuncReference* o);
  virtual void visit(const ProgReference* o);

  virtual void visit(const VarParameter* o);
  virtual void visit(const FuncParameter* o);
  virtual void visit(const ProgParameter* o);

  /**
   * Number of expressions in tuple.
   */
  int size;

  /**
   * Number of range expressions in tuple.
   */
  int dims;
};
}

inline bi::TupleSizer::~TupleSizer() {
  //
}
