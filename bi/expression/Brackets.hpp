/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/common/Unary.hpp"
#include "bi/expression/EmptyExpression.hpp"

namespace bi {
class Parameter;
class VarReference;

/**
 * Expression in brackets.
 *
 * @ingroup compiler_expression
 */
class Brackets: public Expression, public Unary<Expression> {
public:
  /**
   * Constructor.
   *
   * @param single Expression in brackets.
   * @param loc Location.
   */
  Brackets(Expression* single = new EmptyExpression(),
      shared_ptr<Location> loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~Brackets();

  virtual Expression* strip();
  virtual Iterator<Expression> begin() const;
  virtual Iterator<Expression> end() const;

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  using Expression::definitely;
  using Expression::possibly;

  virtual bool dispatchDefinitely(const Expression& o) const;
  virtual bool definitely(const Brackets& o) const;

  virtual bool dispatchPossibly(const Expression& o) const;
  virtual bool possibly(const Brackets& o) const;
};
}
