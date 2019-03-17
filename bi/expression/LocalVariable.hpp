/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/common/Annotated.hpp"
#include "bi/common/Named.hpp"
#include "bi/common/Numbered.hpp"
#include "bi/common/Bracketed.hpp"
#include "bi/common/Argumented.hpp"
#include "bi/common/Valued.hpp"

namespace bi {
/**
 * Local variable.
 *
 * @ingroup expression
 */
class LocalVariable: public Expression,
    public Annotated,
    public Named,
    public Numbered,
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
   * @param value Initial value.
   * @param loc Location.
   */
  LocalVariable(const Annotation annotation, Name* name, Type* type,
      Expression* brackets, Expression* args, Expression* value,
      Location* loc = nullptr);

  /**
   * Constructor for temporary local variable.
   *
   * @param value Initial value.
   * @param loc Location.
   */
  LocalVariable(Expression* value, Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~LocalVariable();

  /**
   * Does this variable need intialization arguments?
   */
  virtual bool needsConstruction() const;

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
