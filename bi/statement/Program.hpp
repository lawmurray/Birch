/**
 * @file
 */
#pragma once

#include "bi/statement/Statement.hpp"
#include "bi/expression/Expression.hpp"
#include "bi/common/Named.hpp"
#include "bi/common/Parenthesised.hpp"
#include "bi/common/Scoped.hpp"
#include "bi/common/Braced.hpp"

namespace bi {
/**
 * Program parameter.
 *
 * @ingroup compiler_program
 */
class Program: public Statement,
    public Named,
    public Parenthesised,
    public Scoped,
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
  Program(shared_ptr<Name> name, Expression* parens, Statement* braces,
      shared_ptr<Location> loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~Program();

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
