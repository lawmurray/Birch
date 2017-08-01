/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/statement/Statement.hpp"
#include "bi/type/Type.hpp"

namespace bi {
/**
 * Object containing another single object.
 *
 * @ingroup compiler_common
 */
template<class T>
class Single {
public:
  /**
   * Constructor.
   *
   * @param single Object.
   */
  Single(T* single);

  /**
   * Destructor.
   */
  virtual ~Single() = 0;

  /**
   * Object.
   */
  T* single;
};
}
