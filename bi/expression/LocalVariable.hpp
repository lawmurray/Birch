/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/common/Named.hpp"
#include "bi/common/Numbered.hpp"
#include "bi/common/Parenthesised.hpp"
#include "bi/common/Valued.hpp"

namespace bi {
/**
 * Local variable.
 *
 * @ingroup compiler_expression
 */
class LocalVariable: public Expression,
    public Named,
    public Numbered,
    public Parenthesised,
    public Valued {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param type Type.
   * @param parens Constructor arguments.
   * @param value Initial value.
   * @param loc Location.
   */
  LocalVariable(Name* name, Type* type, Expression* parens, Expression* value,
      Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~LocalVariable();

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
