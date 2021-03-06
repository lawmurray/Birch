/**
 * @file
 */
#pragma once

#include "src/expression/Expression.hpp"
#include "src/common/Annotated.hpp"
#include "src/common/Named.hpp"
#include "src/common/Numbered.hpp"
#include "src/common/Typed.hpp"
#include "src/common/Valued.hpp"

namespace birch {
/**
 * Parameter to a function, member function, operator, program, or class.
 *
 * @ingroup expression
 */
class Parameter: public Expression,
    public Annotated,
    public Named,
    public Numbered,
    public Typed,
    public Valued {
public:
  /**
   * Constructor.
   *
   * @param annotation Annotation.
   * @param name Name.
   * @param type Type.
   * @param op Initialization operator.
   * @param value Default value.
   * @param loc Location.
   */
  Parameter(const Annotation annotation, Name* name, Type* type, Name* op,
      Expression* value, Location* loc = nullptr);

  virtual void accept(Visitor* visitor) const;
};
}
