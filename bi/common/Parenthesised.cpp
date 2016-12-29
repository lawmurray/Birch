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
  ExpressionList* expr = dynamic_cast<ExpressionList*>(parens->strip());
  assert(expr);
  return expr->head.get();
}

const bi::Expression* bi::Parenthesised::getRight() const {
  ExpressionList* expr = dynamic_cast<ExpressionList*>(parens->strip());
  assert(expr);
  return expr->tail.get();
}
