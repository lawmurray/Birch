/**
 * @file
 */
#pragma once

#include "bi/common/Parenthesised.hpp"

namespace bi {
/**
 * Function form flags.
 */
enum FunctionForm {
  FUNCTION, BINARY_OPERATOR, UNARY_OPERATOR, ASSIGNMENT_OPERATOR, CONSTRUCTOR
};

/**
 * Function that has a form as either a function, binary operator or unary
 * operator.
 *
 * @ingroup compiler_expression
 */
class Formed : public Parenthesised {
public:
  /**
   * Constructor.
   *
   * @param parens Parentheses.
   * @param form Function form.
   */
  Formed(Expression* parens, const FunctionForm form);

  /**
   * Destructor.
   */
  virtual ~Formed() = 0;

  /**
   * Is this a binary operator?
   */
  bool isBinary() const;

  /**
   * Is this a unary operator?
   */
  bool isUnary() const;

  /**
   * Is this an assignment operator?
   */
  bool isAssignment() const;

  /**
   * Is this a constructor?
   */
  bool isConstructor() const;

  /**
   * If these parentheses were constructed for a binary operator, get the
   * left operand. Otherwise undefined.
   */
  const Expression* getLeft() const;

  /**
   * If these parentheses were constructed for a binary or unary operator,
   * get the right operand. Otherwise undefined.
   */
  const Expression* getRight() const;

  /**
   * Form.
   */
  FunctionForm form;
};
}
