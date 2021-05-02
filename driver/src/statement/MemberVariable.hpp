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
#include "src/common/Used.hpp"

namespace birch {
/**
 * Class member variable.
 *
 * @ingroup statement
 */
class MemberVariable: public Statement,
    public Annotated,
    public Named,
    public Numbered,
    public Typed,
    public Bracketed,
    public Argumented,
    public Valued,
    public Used {
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
  MemberVariable(const Annotation annotation, Name* name, Type* type,
      Expression* brackets, Expression* args, Name* op, Expression* value,
      Location* loc = nullptr);

  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
