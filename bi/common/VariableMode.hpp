/**
 * @file
 */
#pragma once

namespace bi {
/**
 * Variable form flags.
 */
enum VariableForm {
  ORDINARY,
  PARAMETER,
  MEMBER
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
  VariableMode(const VariableForm = ORDINARY);

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
