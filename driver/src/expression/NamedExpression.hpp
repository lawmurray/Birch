/**
 * @file
 */
#pragma once

#include "src/expression/Expression.hpp"
#include "src/common/Named.hpp"
#include "src/common/Typed.hpp"
#include "src/common/TypeArgumented.hpp"
#include "src/common/Scope.hpp"
#include "src/common/Used.hpp"

namespace birch {
/**
 * Name in the context of an expression, referring to a variable, function,
 * or operator.
 *
 * @ingroup expression
 */
class NamedExpression:
    public Expression,
    public Named,
    public Typed,
    public TypeArgumented {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param typeArgs Type arguments.
   * @param loc Location.
   */
  NamedExpression(Name* name, Type* typeArgs, Location* loc = nullptr);

  /**
   * Constructor.
   *
   * @param name Name.
   * @param loc Location.
   */
  NamedExpression(Name* name, Location* loc = nullptr);

  virtual bool isAssignable() const;
  virtual bool isGlobal() const;
  virtual bool isMember() const;
  virtual bool isLocal() const;
  virtual bool isParameter() const;

  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  /**
   * Once resolved, the associated variable or function declaration.
   */
  Used* uses;

  /**
   * Once resolved, the rank of this use among all uses of the associated
   * varibale or function declaration.
   */
  int rank;

  /**
   * Once resolved, the unique number of the referent.
   */
  int number;

  /**
   * The category of the identifier.
   */
  ExpressionCategory category;
};
}
