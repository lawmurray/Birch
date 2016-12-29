/**
 * @file
 */
#pragma once

#include "bi/expression/all.hpp"
#include "bi/primitive/unique_ptr.hpp"

namespace bi {
/**
 * Statement with a condition (e.g. conditional, loop).
 *
 * @ingroup compiler_statement
 */
class Conditioned {
public:
  /**
   * Constructor.
   *
   * @param cond Condition.
   */
  Conditioned(Expression* cond);

  /**
   * Destructor.
   */
  virtual ~Conditioned() = 0;

  /**
   * First statement in in brackets.
   */
  unique_ptr<Expression> cond;
};
}

inline bi::Conditioned::Conditioned(Expression* cond) :
    cond(cond) {
  /* pre-condition */
  assert(cond);
}

inline bi::Conditioned::~Conditioned() {
  //
}
