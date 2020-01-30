/**
 * @file
 */
#pragma once

namespace bi {
class Expression;
class Type;

/**
 * Object with generic type parameters.
 *
 * @ingroup common
 */
class TypeParameterised {
public:
  /**
   * Constructor.
   *
   * @param typeParams Generic type parameters.
   */
  TypeParameterised(Expression* typeParams);

  /**
   * Destructor.
   */
  virtual ~TypeParameterised() = 0;

  /**
   * Does this class have generic type parameters?
   */
  bool isGeneric() const;

  /**
   * Generic type parameters.
   */
  Expression* typeParams;
};
}
