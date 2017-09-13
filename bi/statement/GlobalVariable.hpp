/**
 * @file
 */
#pragma once

#include "bi/statement/Statement.hpp"
#include "bi/common/Named.hpp"
#include "bi/common/Numbered.hpp"
#include "bi/common/Typed.hpp"
#include "bi/common/Parenthesised.hpp"
#include "bi/common/Valued.hpp"

namespace bi {
/**
 * Global variable.
 *
 * @ingroup compiler_statement
 */
class GlobalVariable: public Statement,
    public Named,
    public Numbered,
    public Typed,
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
  GlobalVariable(Name* name, Type* type, Expression* parens,
      Expression* value, Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~GlobalVariable();

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
