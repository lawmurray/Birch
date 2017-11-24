/**
 * @file
 */
#pragma once

#include "bi/statement/Statement.hpp"
#include "bi/common/Named.hpp"
#include "bi/common/Numbered.hpp"
#include "bi/common/Typed.hpp"
#include "bi/common/Bracketed.hpp"
#include "bi/common/Argumented.hpp"
#include "bi/common/Valued.hpp"

namespace bi {
/**
 * Global variable.
 *
 * @ingroup birch_statement
 */
class GlobalVariable: public Statement,
    public Named,
    public Numbered,
    public Typed,
    public Bracketed,
    public Argumented,
    public Valued {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param type Type.
   * @param brackets Array size.
   * @param args Constructor arguments.
   * @param value Initial value.
   * @param loc Location.
   */
  GlobalVariable(Name* name, Type* type, Expression* brackets,
      Expression* args, Expression* value, Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~GlobalVariable();

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
