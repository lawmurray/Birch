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
  virtual Expression* modify(ExpressionList* o);
  virtual Statement* modify(StatementList* o);
  virtual Expression* modify(ParenthesesExpression* o);
  virtual Expression* modify(BracesExpression* o);
  virtual Expression* modify(BracketsExpression* o);
  virtual Expression* modify(Index* o);
  virtual Expression* modify(Range* o);
  virtual Expression* modify(Member* o);
  virtual Expression* modify(Super* o);
  virtual Expression* modify(This* o);

  virtual Expression* modify(VarReference* o);
  virtual Expression* modify(FuncReference* o);
  virtual Type* modify(TypeReference* o);
  virtual Prog* modify(ProgReference* o);

  virtual Expression* modify(VarParameter* o);
  virtual Expression* modify(FuncParameter* o);
  virtual Expression* modify(ConversionParameter* o);
  virtual Type* modify(TypeParameter* o);
  virtual Prog* modify(ProgParameter* o);

  virtual void modify(File* o);
  virtual Statement* modify(Import* o);
  virtual Statement* modify(ExpressionStatement* o);
  virtual Statement* modify(If* o);
  virtual Statement* modify(For* o);
  virtual Statement* modify(While* o);
  virtual Statement* modify(Return* o);
  virtual Statement* modify(Raw* o);
  virtual Statement* modify(VarDeclaration* o);
  virtual Statement* modify(FuncDeclaration* o);
  virtual Statement* modify(ConversionDeclaration* o);
  virtual Statement* modify(TypeDeclaration* o);
  virtual Statement* modify(ProgDeclaration* o);

  virtual Type* modify(BracketsType* o);
  virtual Type* modify(ParenthesesType* o);
  virtual Type* modify(LambdaType* o);
  virtual Type* modify(TypeList* o);
};
}
