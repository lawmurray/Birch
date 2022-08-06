/**
 * @file
 */
#pragma once

#include "src/expression/Expression.hpp"
#include "src/statement/Statement.hpp"
#include "src/type/Type.hpp"

namespace birch {
/**
 * Object containing another single object.
 *
 * @ingroup common
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
   * Object.
   */
  T* single;
};
}
