/**
 * @file
 */
#pragma once

#include "bi/visitor/Modifier.hpp"

namespace bi {
/**
 * Finds an object and replaces it with another.
 *
 * @ingroup compiler_visitor
 */
class Replacer: public Modifier {
public:
  /**
   * Constructor.
   *
   * @param find Object to find.
   * @param replace Object to replace it with.
   */
  Replacer(Expression* find, Expression* replace);

  /**
   * Destructor.
   */
  virtual ~Replacer();

  using Modifier::modify;
  virtual Expression* modify(EmptyExpression* o);
  virtual Expression* modify(BooleanLiteral* o);
  virtual Expression* modify(IntegerLiteral* o);
  virtual Expression* modify(RealLiteral* o);
  virtual Expression* modify(StringLiteral* o);
  virtual Expression* modify(ExpressionList* o);
  virtual Expression* modify(ParenthesesExpression* o);
  virtual Expression* modify(BracesExpression* o);
  virtual Expression* modify(BracketsExpression* o);
  virtual Expression* modify(Index* o);
  virtual Expression* modify(Range* o);
  virtual Expression* modify(Member* o);
  virtual Expression* modify(This* o);
  virtual Expression* modify(LambdaInit* o);
  virtual Expression* modify(RandomInit* o);
  virtual Expression* modify(VarReference* o);
  virtual Expression* modify(FuncReference* o);
  virtual Expression* modify(VarParameter* o);
  virtual Expression* modify(FuncParameter* o);

protected:
  /**
   * Object to find.
   */
  Expression* find;

  /**
   * Object to replace it with.
   */
  Expression* replace;
};
}
