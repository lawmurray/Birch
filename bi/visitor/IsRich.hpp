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

  virtual void visit(const EmptyExpression* o);
  virtual void visit(const EmptyStatement* o);
  virtual void visit(const BooleanLiteral* o);
  virtual void visit(const IntegerLiteral* o);
  virtual void visit(const RealLiteral* o);
  virtual void visit(const StringLiteral* o);
  virtual void visit(const Name* o);
  virtual void visit(const Path* o);
  virtual void visit(const ExpressionList* o);
  virtual void visit(const StatementList* o);
  virtual void visit(const TypeList* o);
  virtual void visit(const ParenthesesExpression* o);
  virtual void visit(const BracesExpression* o);
  virtual void visit(const BracketsExpression* o);
  virtual void visit(const Range* o);
  virtual void visit(const Member* o);
  virtual void visit(const This* o);
  virtual void visit(const RandomInit* o);

  virtual void visit(const VarReference* o);
  virtual void visit(const FuncReference* o);
  virtual void visit(const ModelReference* o);

  virtual void visit(const VarParameter* o);
  virtual void visit(const FuncParameter* o);
  virtual void visit(const ModelParameter* o);
  virtual void visit(const ProgParameter* o);

  virtual void visit(const File* o);
  virtual void visit(const Import* o);
  virtual void visit(const ExpressionStatement* o);
  virtual void visit(const Conditional* o);
  virtual void visit(const Loop* o);
  virtual void visit(const Raw* o);
  virtual void visit(const FuncDeclaration* o);
  virtual void visit(const ModelDeclaration* o);
  virtual void visit(const ProgDeclaration* o);

  /**
   * Result.
   */
  bool result;
};
}
