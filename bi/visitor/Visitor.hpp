/**
 * @file
 */
#pragma once

#include "bi/expression/all.hpp"
#include "bi/expression/all.hpp"
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

  virtual void visit(const Name* o);
  virtual void visit(const Path* o);

  virtual void visit(const EmptyExpression* o);
  virtual void visit(const EmptyStatement* o);
  virtual void visit(const EmptyType* o);

  virtual void visit(const BooleanLiteral* o);
  virtual void visit(const IntegerLiteral* o);
  virtual void visit(const RealLiteral* o);
  virtual void visit(const StringLiteral* o);
  virtual void visit(const List<Expression>* o);
  virtual void visit(const List<Statement>* o);
  virtual void visit(const ParenthesesExpression* o);
  virtual void visit(const BracesExpression* o);
  virtual void visit(const BracketsExpression* o);
  virtual void visit(const Index* o);
  virtual void visit(const Span* o);
  virtual void visit(const Range* o);
  virtual void visit(const Member* o);
  virtual void visit(const Super* o);
  virtual void visit(const This* o);

  virtual void visit(const VarReference* o);
  virtual void visit(const FuncReference* o);
  virtual void visit(const BinaryReference* o);
  virtual void visit(const UnaryReference* o);
  virtual void visit(const AssignmentReference* o);
  virtual void visit(const TypeReference* o);
  virtual void visit(const ProgReference* o);

  virtual void visit(const VarParameter* o);
  virtual void visit(const FuncParameter* o);
  virtual void visit(const BinaryParameter* o);
  virtual void visit(const UnaryParameter* o);
  virtual void visit(const AssignmentParameter* o);
  virtual void visit(const ConversionParameter* o);
  virtual void visit(const TypeParameter* o);
  virtual void visit(const ProgParameter* o);

  virtual void visit(const File* o);
  virtual void visit(const Import* o);
  virtual void visit(const ExpressionStatement* o);
  virtual void visit(const If* o);
  virtual void visit(const For* o);
  virtual void visit(const While* o);
  virtual void visit(const Return* o);
  virtual void visit(const Raw* o);
  virtual void visit(const Declaration<Expression>* o);
  virtual void visit(const Declaration<Type>* o);

  virtual void visit(const BracketsType* o);
  virtual void visit(const ParenthesesType* o);
  virtual void visit(const FunctionType* o);
  virtual void visit(const CoroutineType* o);
  virtual void visit(const List<Type>* o);
};
}
