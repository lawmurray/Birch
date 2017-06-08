/**
 * @file
 */
#pragma once

namespace bi {
/**
 * Function form flags.
 */
enum FunctionForm {
  FUNCTION,
  BINARY_OPERATOR,
  UNARY_OPERATOR,
  ASSIGN_OPERATOR,
  LAMBDA_FUNCTION,
  MEMBER_FUNCTION
};

/**
 * Function mode.
 *
 * @ingroup compiler_expression
 */
class FunctionMode {
public:
  /**
   * Constructor.
   *
   * @param form Form.
   */
  FunctionMode(const FunctionForm form);

  /**
   * Destructor.
   */
  virtual ~FunctionMode() = 0;

  /**
   * Is this an operator?
   */
  bool isOperator() const;

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
  bool isAssign() const;

  /**
   * Is this a lambda function?
   */
  bool isLambda() const;

  /**
   * Is this a lambda function?
   */
  bool isMember() const;

  /**
   * Form.
   */
  FunctionForm form;
};
}
