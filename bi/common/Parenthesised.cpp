/**
 * @file
 */
#include "bi/common/Parenthesised.hpp"

#include "bi/common/List.hpp"

bi::Parenthesised::Parenthesised(Expression* parens) :
    parens(parens) {
  /* pre-condition */
  assert(parens);
}

bi::Parenthesised::~Parenthesised() {
  //
}

const bi::Expression* bi::Parenthesised::getLeft() const {
  Expression* strip = parens->strip();
  ExpressionList* expr = dynamic_cast<ExpressionList*>(strip);
  assert(expr);
  return expr->head.get();
}

bi::Expression* bi::Parenthesised::releaseLeft() {
  Expression* strip = parens->strip();
  ExpressionList* expr = dynamic_cast<ExpressionList*>(strip);
  assert(expr);
  return expr->head.release();
}

const bi::Expression* bi::Parenthesised::getRight() const {
  Expression* strip = parens->strip();
  ExpressionList* expr = dynamic_cast<ExpressionList*>(strip);
  assert(expr);
  return expr->tail.get();
}

bi::Expression* bi::Parenthesised::releaseRight() {
  Expression* strip = parens->strip();
  ExpressionList* expr = dynamic_cast<ExpressionList*>(strip);
  assert(expr);
  return expr->tail.release();
}
