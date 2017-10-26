/**
 * @file
 */
#pragma once

#include "bi/statement/Statement.hpp"
#include "bi/common/Named.hpp"
#include "bi/common/Numbered.hpp"
#include "bi/common/Typed.hpp"
#include "bi/common/Argumented.hpp"
#include "bi/common/Valued.hpp"

namespace bi {
/**
 * Class member variable.
 *
 * @ingroup compiler_statement
 */
class MemberVariable: public Statement,
    public Named,
    public Numbered,
    public Typed,
    public Argumented,
    public Valued {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param type Type.
   * @param args Constructor arguments.
   * @param value Initial value.
   * @param loc Location.
   */
  MemberVariable(Name* name, Type* type, Expression* args,
      Expression* value, Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~MemberVariable();

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
