/**
 * @file
 */
#pragma once

#include "bi/type/Type.hpp"
#include "bi/type/EmptyType.hpp"
#include "bi/primitive/unique_ptr.hpp"

namespace bi {
/**
 * Typed expression or statement.
 *
 * @ingroup compiler_common
 */
class Typed {
public:
  /**
   * Constructor.
   *
   * @param type Type.
   */
  Typed(Type* type = new EmptyType());

  /**
   * Destructor.
   */
  virtual ~Typed() = 0;

  /**
   * Type.
   */
  unique_ptr<Type> type;
};
}
