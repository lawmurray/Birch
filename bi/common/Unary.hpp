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
  Unary(T* single);

  /**
   * Destructor.
   */
  virtual ~Unary() = 0;

  /**
   * Operand.
   */
  unique_ptr<T> single;
};
}
