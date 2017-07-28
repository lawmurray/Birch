/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/common/Unary.hpp"
#include "bi/common/Parenthesised.hpp"
#include "bi/common/Reference.hpp"
#include "bi/expression/Parameter.hpp"

namespace bi {
/**
 * Call to a function.
 *
 * @ingroup compiler_expression
 */
template<class ObjectType>
class OverloadedCall: public Expression,
    public Unary<Expression>,
    public Parenthesised,
    public Reference<ObjectType> {
public:
  /**
   * Constructor.
   *
   * @param single Expression.
   * @param parens Parentheses.
   * @param loc Location.
   * @param target Target.
   */
  OverloadedCall(Expression* single, Expression* parens,
      shared_ptr<Location> loc = nullptr, const ObjectType* target = nullptr);

  /**
   * Destructor.
   */
  virtual ~OverloadedCall();

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  using Expression::definitely;
  using Expression::possibly;

  virtual bool dispatchDefinitely(const Expression& o) const;
  virtual bool definitely(const OverloadedCall<ObjectType>& o) const;
  virtual bool definitely(const ObjectType& o) const;
  virtual bool definitely(const Parameter& o) const;

  virtual bool dispatchPossibly(const Expression& o) const;
  virtual bool possibly(const OverloadedCall<ObjectType>& o) const;
  virtual bool possibly(const ObjectType& o) const;
  virtual bool possibly(const Parameter& o) const;
};
}
