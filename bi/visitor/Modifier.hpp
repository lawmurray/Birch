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
 * Modifying visitor.
 *
 * @ingroup compiler_visitor
 */
class Modifier {
public:
  /**
   * Destructor.
   */
  virtual ~Modifier();

  virtual Expression* modify(EmptyExpression* o);
  virtual Statement* modify(EmptyStatement* o);
  virtual Type* modify(EmptyType* o);

  virtual Expression* modify(BooleanLiteral* o);
  virtual Expression* modify(IntegerLiteral* o);
  virtual Expression* modify(RealLiteral* o);
  virtual Expression* modify(StringLiteral* o);
  virtual Expression* modify(List<Expression>* o);
  virtual Statement* modify(List<Statement>* o);
  virtual Expression* modify(ParenthesesExpression* o);
  virtual Expression* modify(BracesExpression* o);
  virtual Expression* modify(BracketsExpression* o);
  virtual Expression* modify(Span* o);
  virtual Expression* modify(Index* o);
  virtual Expression* modify(Range* o);
  virtual Expression* modify(Member* o);
  virtual Expression* modify(Super* o);
  virtual Expression* modify(This* o);

  virtual Expression* modify(VarReference* o);
  virtual Expression* modify(FuncReference* o);
  virtual Expression* modify(BinaryReference* o);
  virtual Expression* modify(UnaryReference* o);
  virtual Statement* modify(AssignmentReference* o);
  virtual Type* modify(TypeReference* o);

  virtual Expression* modify(VarParameter* o);
  virtual Statement* modify(FuncParameter* o);
  virtual Statement* modify(BinaryParameter* o);
  virtual Statement* modify(UnaryParameter* o);
  virtual Statement* modify(AssignmentParameter* o);
  virtual Statement* modify(ConversionParameter* o);
  virtual Statement* modify(ProgParameter* o);
  virtual Type* modify(TypeParameter* o);

  virtual void modify(File* o);
  virtual Statement* modify(Import* o);
  virtual Statement* modify(ExpressionStatement* o);
  virtual Statement* modify(If* o);
  virtual Statement* modify(For* o);
  virtual Statement* modify(While* o);
  virtual Statement* modify(Return* o);
  virtual Statement* modify(Raw* o);

  virtual Type* modify(BracketsType* o);
  virtual Type* modify(ParenthesesType* o);
  virtual Type* modify(FunctionType* o);
  virtual Type* modify(CoroutineType* o);
  virtual Type* modify(List<Type>* o);
};
}
