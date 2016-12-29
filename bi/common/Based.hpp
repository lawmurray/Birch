/**
 * @file
 */
#pragma once

#include "bi/type/Type.hpp"
#include "bi/common/Name.hpp"
#include "bi/primitive/unique_ptr.hpp"

namespace bi {
/**
 * Type with base.
 *
 * @ingroup compiler_type
 */
class Based {
public:
  /**
   * Constructor.
   *
   * @param op Operator giving relation to base type.
   * @param base Base type.
   */
  Based(shared_ptr<Name> op, Type* base);

  /**
   * Destructor.
   */
  virtual ~Based() = 0;

  /**
   * Operator giving relation to base type.
   */
  shared_ptr<Name> op;

  /**
   * Base type.
   */
  unique_ptr<Type> base;
};
}
