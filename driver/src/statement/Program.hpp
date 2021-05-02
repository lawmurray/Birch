/**
 * @file
 */
#pragma once

#include "src/statement/Statement.hpp"
#include "src/expression/Expression.hpp"
#include "src/common/Named.hpp"
#include "src/common/Numbered.hpp"
#include "src/common/Parameterised.hpp"
#include "src/common/Scoped.hpp"
#include "src/common/Braced.hpp"

namespace birch {
/**
 * Program parameter.
 *
 * @ingroup statement
 */
class Program: public Statement,
    public Named,
    public Numbered,
    public Parameterised,
    public Scoped,
    public Braced {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param params Parameters.
   * @param braces Body.
   * @param loc Location.
   */
  Program(Name* name, Expression* params, Statement* braces,
      Location* loc = nullptr);

  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
