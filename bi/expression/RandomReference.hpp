/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/expression/RandomParameter.hpp"
#include "bi/common/Reference.hpp"

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


  virtual Expression* acceptClone(Cloner* visitor) const;
  virtual Expression* acceptModify(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual bool operator<=(Expression& o);
  virtual bool operator==(const Expression& o) const;

  /**
   * Reference for the variate.
   */
  unique_ptr<Expression> x;

  /**
   * Reference for the model.
   */
  unique_ptr<Expression> m;
};
}
