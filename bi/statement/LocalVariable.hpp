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
#include "bi/primitive/unique_ptr.hpp"

namespace bi {
/**
 * Global variable.
 *
 * @ingroup compiler_expression
 */
class LocalVariable: public Statement,
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
  LocalVariable(shared_ptr<Name> name, Type* type, Expression* parens =
      new EmptyExpression(), Expression* value = new EmptyExpression(),
      shared_ptr<Location> loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~LocalVariable();

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  using Statement::definitely;
  using Statement::possibly;

  virtual bool dispatchDefinitely(const Statement& o) const;
  virtual bool definitely(const LocalVariable& o) const;

  virtual bool dispatchPossibly(const Statement& o) const;
  virtual bool possibly(const LocalVariable& o) const;
};
}
