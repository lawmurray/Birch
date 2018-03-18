/**
 * @file
 */
#pragma once

#include "bi/visitor/Resolver.hpp"

namespace bi {
/**
 * This is the third pass of the abstract syntax tree after parsing,
 * populating available variables, functions, etc (everything other than
 * types).
 *
 * @ingroup visitor
 */
class ResolverHeader: public Resolver {
public:
  /**
   * Constructor.
   *
   * @param rootScope The root scope.
   */
  ResolverHeader(Scope* rootScope);

  /**
   * Destructor.
   */
  virtual ~ResolverHeader();

  using Resolver::modify;

  virtual Expression* modify(Parameter* o);
  virtual Expression* modify(Generic* o);

  virtual Statement* modify(Basic* o);
  virtual Statement* modify(Class* o);
  virtual Statement* modify(Explicit* o);
  virtual Statement* modify(GlobalVariable* o);
  virtual Statement* modify(Function* o);
  virtual Statement* modify(Fiber* o);
  virtual Statement* modify(Program* o);
  virtual Statement* modify(BinaryOperator* o);
  virtual Statement* modify(UnaryOperator* o);
  virtual Statement* modify(MemberVariable* o);
  virtual Statement* modify(MemberFunction* o);
  virtual Statement* modify(MemberFiber* o);
  virtual Statement* modify(AssignmentOperator* o);
  virtual Statement* modify(ConversionOperator* o);
};
}
