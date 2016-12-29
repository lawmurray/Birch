/**
 * @file
 */
#pragma once

#include "bi/expression/all.hpp"
#include "bi/program/all.hpp"
#include "bi/statement/all.hpp"
#include "bi/type/all.hpp"

namespace bi {
/**
 * Visitor.
 *
 * @ingroup compiler_visitor
 */
class Visitor {
public:
  /**
   * Destructor.
   */
  virtual ~Visitor();

  virtual void visit(const EmptyExpression* o);
  virtual void visit(const EmptyStatement* o);
  virtual void visit(const EmptyType* o);

  virtual void visit(const BoolLiteral* o);
  virtual void visit(const IntLiteral* o);
  virtual void visit(const RealLiteral* o);
  virtual void visit(const StringLiteral* o);
  virtual void visit(const Name* o);
  virtual void visit(const Path* o);
  virtual void visit(const ExpressionList* o);
  virtual void visit(const StatementList* o);
  virtual void visit(const ParenthesesExpression* o);
  virtual void visit(const BracesExpression* o);
  virtual void visit(const Range* o);
  virtual void visit(const Traversal* o);
  virtual void visit(const This* o);
  virtual void visit(const BracketsExpression* o);

  virtual void visit(const VarReference* o);
  virtual void visit(const FuncReference* o);
  virtual void visit(const ModelReference* o);
  virtual void visit(const ProgReference* o);

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
  virtual void visit(const VarDeclaration* o);
  virtual void visit(const FuncDeclaration* o);
  virtual void visit(const ModelDeclaration* o);
  virtual void visit(const ProgDeclaration* o);

  virtual void visit(const ParenthesesType* o);
  virtual void visit(const TypeList* o);
};
}

inline bi::Visitor::~Visitor() {
  //
}
