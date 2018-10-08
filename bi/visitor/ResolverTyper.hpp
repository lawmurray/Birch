/**
 * @file
 */
#pragma once

#include "bi/visitor/Modifier.hpp"

namespace bi {
/**
 * This is the first pass of the abstract syntax tree after parsing,
 * populating available types. This is analogous to having forward
 * declarations of all types as in C++.
 *
 * @ingroup visitor
 */
class ResolverTyper: public Modifier {
public:
  /**
   * Constructor.
   *
   * @param rootScope The root scope.
   */
  ResolverTyper(Scope* rootScope);

  /**
   * Destructor.
   */
  virtual ~ResolverTyper();

  using Modifier::modify;

  virtual Expression* modify(Generic* o);
  virtual Statement* modify(Basic* o);
  virtual Statement* modify(Class* o);
  virtual Statement* modify(Function* o);
  virtual Statement* modify(Fiber* o);

protected:
  /**
   * List of scopes, innermost at the back.
   */
  std::list<Scope*> scopes;
};
}
