/**
 * @file
 */
#pragma once

#include "bi/statement/Statement.hpp"
#include "bi/expression/Expression.hpp"
#include "bi/common/Named.hpp"
#include "bi/common/Scoped.hpp"
#include "bi/common/Parenthesised.hpp"
#include "bi/common/Braced.hpp"

namespace bi {
/**
 * Program parameter.
 *
 * @ingroup compiler_program
 */
class ProgParameter: public Statement,
    public Named,
    public Scoped,
    public Parenthesised,
    public Braced {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param parens Expression in parentheses.
   * @param braces Expression in braces.
   * @param loc Location.
   */
  ProgParameter(shared_ptr<Name> name, Expression* parens, Expression* braces,
      shared_ptr<Location> loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~ProgParameter();

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  using Statement::definitely;
  using Statement::possibly;

  virtual bool dispatchDefinitely(const Statement& o) const;
  virtual bool definitely(const ProgParameter& o) const;

  virtual bool dispatchPossibly(const Statement& o) const;
  virtual bool possibly(const ProgParameter& o) const;
};
}
