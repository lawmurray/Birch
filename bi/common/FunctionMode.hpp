/**
 * @file
 */
#pragma once

namespace bi {
/**
 * Function form flags.
 */
enum FunctionForm {
  ASSIGN_FORM,
  FUNCTION_FORM,
  LAMBDA_FUNCTION_FORM,
  MEMBER_FUNCTION_FORM,
  COROUTINE_FORM,
  LAMBDA_COROUTINE_FORM,
  MEMBER_COROUTINE_FORM,
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
  FunctionMode(const FunctionForm form = FUNCTION_FORM);

  /**
   * Destructor.
   */
  virtual ~FunctionMode() = 0;

  /**
   * Is this an operator?
   */
  bool isOperator() const;

  /**
   * Is this an assignment operator?
   */
  bool isAssign() const;

  /**
   * Is this a lambda?
   */
  bool isLambda() const;

  /**
   * Is this a member?
   */
  bool isMember() const;

  /**
   * Is this a function?
   */
  bool isFunction() const;

  /**
   * Is this a coroutine?
   */
  bool isCoroutine() const;

  /**
   * Form.
   */
  FunctionForm form;
};
}
