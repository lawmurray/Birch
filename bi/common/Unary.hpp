/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/statement/Statement.hpp"
#include "bi/type/Type.hpp"
#include "bi/primitive/unique_ptr.hpp"

namespace bi {
/**
 * Object containing single operand.
 *
 * @ingroup compiler_common
 */
template<class T>
class Unary {
public:
  /**
   * Constructor.
   *
   * @param single Operand.
   */
  explicit Unary(T* single);

  /**
   * Destructor.
   */
  virtual ~Unary() = 0;

  /**
   * Operand.
   */
  unique_ptr<T> single;
};

/**
 * Expression unary.
 *
 * @ingroup compiler_common
 */
typedef Unary<Expression> ExpressionUnary;

/**
 * Statement unary.
 *
 * @ingroup compiler_common
 */
typedef Unary<Statement> StatementUnary;

/**
 * Type unnary.
 *
 * @ingroup compiler_common
 */
typedef Unary<Type> TypeUnary;

}
