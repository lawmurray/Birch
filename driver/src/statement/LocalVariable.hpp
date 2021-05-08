/**
 * @file
 */
#pragma once

#include "src/statement/Statement.hpp"
#include "src/common/Annotated.hpp"
#include "src/common/Named.hpp"
#include "src/common/Numbered.hpp"
#include "src/common/Typed.hpp"
#include "src/common/Bracketed.hpp"
#include "src/common/Argumented.hpp"
#include "src/common/Valued.hpp"

namespace birch {
/**
 * Local variable.
 *
 * @ingroup statement
 */
class LocalVariable: public Statement,
    public Annotated,
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
   * @param annotation Annotation.
   * @param name Name.
   * @param type Type.
   * @param brackets Array size.
   * @param args Constructor arguments.
   * @param op Initialization operator.
   * @param value Initial value.
   * @param loc Location.
   */
  LocalVariable(const Annotation annotation, Name* name, Type* type,
      Expression* brackets, Expression* args, Name* op, Expression* value,
      Location* loc = nullptr);

  /**
   * Constructor for temporary local variable.
   *
   * @param value Initial value.
   * @param loc Location.
   */
  LocalVariable(Expression* value, Location* loc = nullptr);

  /**
   * Constructor for loop index variable.
   *
   * @param annotation Annotation.
   * @param name Name.
   * @param type Type.
   * @param loc Location.
   */
  LocalVariable(Name* name, Type* type, Location* loc = nullptr);

  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
