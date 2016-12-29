/**
 * @file
 */
#pragma once

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
class Formed {
public:
  /**
   * Constructor.
   *
   * @param op Operator giving relation to base type.
   * @param base Base type.
   */
  Formed(const FunctionForm form);

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
   * Form.
   */
  FunctionForm form;
};
}
