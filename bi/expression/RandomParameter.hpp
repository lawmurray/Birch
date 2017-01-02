/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/common/Named.hpp"
#include "bi/common/Binary.hpp"
#include "bi/common/Parameter.hpp"
#include "bi/expression/FuncReference.hpp"

namespace bi {
/**
 * Random variable expression.
 *
 * @ingroup compiler_expression
 */
class RandomParameter: public Expression,
    public Named,
    public ExpressionBinary,
    public Parameter<Expression> {
public:
  /**
   * Constructor.
   *
   * @param left Left operand.
   * @param right Right operand.
   * @param loc Location.
   */
  RandomParameter(Expression* left, Expression* right,
      shared_ptr<Location> loc = nullptr);

  /**
   * Constructor.
   */
  RandomParameter(FuncReference* ref);

  /**
   * Destructor.
   */
  virtual ~RandomParameter();

  virtual Expression* acceptClone(Cloner* visitor) const;
  virtual Expression* acceptModify(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual bool operator<=(Expression& o);
  virtual bool operator==(const Expression& o) const;

  /**
   * Pull method.
   */
  unique_ptr<Expression> pull;

  /**
   * Push method.
   */
  unique_ptr<Expression> push;
};
}
