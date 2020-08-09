/**
 * @file
 */
#pragma once

namespace bi {
/**
 * Object containing two other objects, denoted left and right.
 *
 * @ingroup common
 */
template<class T>
class Couple {
public:
  /**
   * Constructor.
   *
   * @param left Left.
   * @param right Right.
   */
  Couple(T* left, T* right);

  /**
   * Destructor.
   */
  virtual ~Couple() = 0;

  /**
   * Left.
   */
  T* left;

  /**
   * Right.
   */
  T* right;
};
}
