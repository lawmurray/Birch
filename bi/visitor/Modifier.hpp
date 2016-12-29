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

  virtual void modify(EmptyExpression* o);
  virtual void modify(EmptyStatement* o);
  virtual void modify(EmptyType* o);

  virtual void modify(BoolLiteral* o);
  virtual void modify(IntLiteral* o);
  virtual void modify(RealLiteral* o);
  virtual void modify(StringLiteral* o);
  virtual void modify(Name* o);
  virtual void modify(Path* o);
  virtual void modify(ExpressionList* o);
  virtual void modify(StatementList* o);
  virtual void modify(ParenthesesExpression* o);
  virtual void modify(BracesExpression* o);
  virtual void modify(BracketsExpression* o);
  virtual void modify(Range* o);
  virtual void modify(Traversal* o);
  virtual void modify(This* o);

  virtual void modify(VarReference* o);
  virtual void modify(FuncReference* o);
  virtual void modify(ModelReference* o);
  virtual void modify(ProgReference* o);

  virtual void modify(VarParameter* o);
  virtual void modify(FuncParameter* o);
  virtual void modify(ModelParameter* o);
  virtual void modify(ProgParameter* o);

  virtual void modify(File* o);
  virtual void modify(Import* o);
  virtual void modify(ExpressionStatement* o);
  virtual void modify(Conditional* o);
  virtual void modify(Loop* o);
  virtual void modify(Raw* o);
  virtual void modify(VarDeclaration* o);
  virtual void modify(FuncDeclaration* o);
  virtual void modify(ModelDeclaration* o);
  virtual void modify(ProgDeclaration* o);

  virtual void modify(ParenthesesType* o);
  virtual void modify(TypeList* o);
};
}

inline bi::Modifier::~Modifier() {
  //
}
