/**
 * @file
 */
#pragma once

#include "bi/statement/Statement.hpp"
#include "bi/common/Annotated.hpp"
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
 * @ingroup statement
 */
class GlobalVariable: public Statement,
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
   * @param value Initial value.
   * @param loc Location.
   */
  GlobalVariable(const Annotation annotation, Name* name, Type* type,
      Expression* brackets, Expression* args, Expression* value,
      Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~GlobalVariable();

  /**
   * Does this variable need intialization arguments?
   */
  virtual bool needsConstruction() const;

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
