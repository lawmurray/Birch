/**
 * @file
 */
#pragma once

namespace birch {
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
   * Left.
   */
  T* left;

  /**
   * Right.
   */
  T* right;
};
}
