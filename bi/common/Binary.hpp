/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"

namespace bi {
/**
 * Object with left and right operands.
 *
 * @ingroup compiler_common
 */
template<class T>
class Binary {
public:
  /**
   * Constructor.
   *
   * @param left Left operand.
   * @param right Right operand.
   */
  Binary(T* left, T* right);

  /**
   * Destructor.
   */
  virtual ~Binary() = 0;

  /**
   * Left operand.
   */
  T* left;

  /**
   * Right operand.
   */
  T* right;
};
}
