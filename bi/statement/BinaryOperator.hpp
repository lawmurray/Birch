/**
 * @file
 */
#pragma once

#include "bi/statement/Statement.hpp"
#include "bi/expression/Expression.hpp"
#include "bi/common/Named.hpp"
#include "bi/common/Numbered.hpp"
#include "bi/common/Parenthesised.hpp"
#include "bi/common/ReturnTyped.hpp"
#include "bi/common/Scoped.hpp"
#include "bi/common/Braced.hpp"

namespace bi {
/**
 * Binary operator.
 *
 * @ingroup compiler_statement
 */
class BinaryOperator: public Statement,
    public Named,
    public Numbered,
    public Parenthesised,
    public ReturnTyped,
    public Typed,
    public Scoped,
    public Braced {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param parens Parentheses.
   * @param returnType Return type.
   * @param braces Braces expression.
   * @param loc Location.
   */
  BinaryOperator(shared_ptr<Name> name, Expression* parens, Type* returnType,
      Statement* braces, shared_ptr<Location> loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~BinaryOperator();

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
