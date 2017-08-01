/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/common/Single.hpp"
#include "bi/common/Argumented.hpp"

namespace bi {
/**
 * Call to a function object. Also used as a placeholder during parsing for
 * calls to first-class overloadable functions before resolution.
 *
 * @ingroup compiler_expression
 */
class Call: public Expression, public Single<Expression>, public Argumented {
public:
  /**
   * Constructor.
   *
   * @param single Expression.
   * @param args Arguments.
   * @param loc Location.
   */
  Call(Expression* single, Expression* args, Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~Call();

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  /**
   * Arguments.
   */
  Expression* args;
};
}
