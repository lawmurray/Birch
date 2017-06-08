/**
 * @file
 */
#pragma once

namespace bi {
/**
 * Variable form flags.
 */
enum VariableForm {
  VARIABLE_FORM,
  PARAMETER_FORM,
  MEMBER_VARIABLE_FORM
};

/**
 * Variable mode.
 *
 * @ingroup compiler_expression
 */
class VariableMode {
public:
  /**
   * Constructor.
   *
   * @param form Form.
   */
  VariableMode(const VariableForm = VARIABLE_FORM);

  /**
   * Destructor.
   */
  virtual ~VariableMode() = 0;

  /**
   * Is this a function parameter?
   */
  bool isParameter() const;

  /**
   * Is this a class member?
   */
  bool isMember() const;

  /**
   * Form.
   */
  VariableForm form;
};
}
