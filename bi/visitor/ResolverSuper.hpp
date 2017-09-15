/**
 * @file
 */
#pragma once

#include "bi/visitor/Resolver.hpp"

namespace bi {
/**
 * This is the second pass of the abstract syntax tree after parsing,
 * establishing super type and conversion type relationships.
 *
 * @ingroup compiler_visitor
 */
class ResolverSuper: public Resolver {
public:
  /**
   * Constructor.
   */
  ResolverSuper();

  /**
   * Destructor.
   */
  virtual ~ResolverSuper();

  using Resolver::modify;

  virtual Statement* modify(ConversionOperator* o);
  virtual Statement* modify(Class* o);
  virtual Statement* modify(Alias* o);
};
}
