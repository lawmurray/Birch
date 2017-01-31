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
