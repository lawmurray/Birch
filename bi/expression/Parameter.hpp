/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/common/Annotated.hpp"
#include "bi/common/Named.hpp"
#include "bi/common/Numbered.hpp"
#include "bi/common/Typed.hpp"
#include "bi/common/Valued.hpp"

namespace bi {
/**
 * Parameter to a function, member function, operator, program, or class.
 *
 * @ingroup statement
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
   * @param value Default value.
   * @param loc Location.
   */
  Parameter(const Annotation annotation, Name* name, Type* type,
      Expression* value, Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~Parameter();

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
