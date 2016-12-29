/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/expression/EmptyExpression.hpp"
#include "bi/primitive/unique_ptr.hpp"

#include <cassert>

namespace bi {
/**
 * Object with brackets.
 *
 * @ingroup compiler_common
 */
class Bracketed {
public:
  /**
   * Constructor.
   *
   * @param brackets Expression in square brackets.
   */
  Bracketed(Expression* brackets = new EmptyExpression());

  /**
   * Destructor.
   */
  virtual ~Bracketed() = 0;

  /**
   * Square bracket expression.
   */
  unique_ptr<Expression> brackets;
};
}

inline bi::Bracketed::Bracketed(Expression* brackets) :
    brackets(brackets) {
  /* pre-condition */
  assert(brackets);
}

inline bi::Bracketed::~Bracketed() {
  //
}
