/**
 * @file
 */
#pragma once

#include "bi/visitor/Cloner.hpp"

namespace bi {
/**
 * Visitor to instantiate a class with generic type parameters.
 *
 * @ingroup compiler_visitor
 */
class Instantiater : public Cloner {
public:
  /**
   * Constructor.
   *
   * @param typeParams Type parameters.
   * @param typeArgs Type arguments.
   */
  Instantiater(const Statement* typeParams, const Type* typeArgs);

  /**
   * Destructor.
   */
  virtual ~Instantiater();

  using Cloner::clone;

protected:
  /**
   * Mapping of substituations, from type parameter names to type arguments.
   */
  std::unordered_map<std::string,const Type*> substitutions;
};
}
