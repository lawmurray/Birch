/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/common/Named.hpp"
#include "bi/common/Binary.hpp"
#include "bi/common/Parameter.hpp"
#include "bi/expression/FuncReference.hpp"
#include "bi/expression/VarParameter.hpp"
#include "bi/expression/VarReference.hpp"

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

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  /**
   * Pull method.
   */
  unique_ptr<Expression> pull;

  /**
   * Push method.
   */
  unique_ptr<Expression> push;

  virtual bool dispatch(Expression& o);
  virtual bool le(RandomParameter& o);
  virtual bool le(VarParameter& o);
  virtual bool le(VarReference& o);
};
}
