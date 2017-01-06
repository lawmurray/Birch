/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/expression/RandomParameter.hpp"
#include "bi/common/Reference.hpp"
#include "bi/expression/VarParameter.hpp"

namespace bi {
/**
 * Reference to a random variable.
 *
 * @ingroup compiler_expression
 */
class RandomReference: public Expression, public Named, public Reference<
    RandomParameter> {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param loc Location.
   * @param target Target.
   */
  RandomReference(shared_ptr<Name> name, shared_ptr<Location> loc = nullptr,
      const RandomParameter* target = nullptr);

  /**
   * Construct from expression.
   *
   * @param expr Expression.
   */
  RandomReference(Expression* expr);

  /**
   * Destructor.
   */
  virtual ~RandomReference();

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  /**
   * Right (model) side of the random variable.
   */
  unique_ptr<Expression> right;

  virtual bool dispatch(Expression& o);
  virtual bool le(RandomParameter& o);
  virtual bool le(RandomReference& o);
  virtual bool le(VarParameter& o);
};
}
