/**
 * @file
 */
#pragma once

#include "bi/expression/BracesExpression.hpp"
#include "bi/common/Scope.hpp"
#include "bi/primitive/unique_ptr.hpp"

namespace bi {
/**
 * Object with braces.
 *
 * @ingroup compiler_common
 */
class Braced {
public:
  /**
   * Constructor.
   *
   * @param braces Body.
   */
  Braced(Expression* braces);

  /**
   * Destructor.
   */
  virtual ~Braced() = 0;

  /**
   * Body.
   */
  unique_ptr<Expression> braces;
};
}

inline bi::Braced::Braced(Expression* braces) :
    braces(braces) {
  /* pre-condition */
  assert(braces);
}

inline bi::Braced::~Braced() {
  //
}
