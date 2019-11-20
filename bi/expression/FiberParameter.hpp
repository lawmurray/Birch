/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/common/Annotated.hpp"
#include "bi/common/Named.hpp"
#include "bi/common/Numbered.hpp"
#include "bi/common/Valued.hpp"

namespace bi {
/**
 * Parameter to a fiber or member fiber.
 *
 * @ingroup statement
 */
class FiberParameter: public Expression,
    public Annotated,
    public Named,
    public Numbered,
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
  FiberParameter(const Annotation annotation, Name* name, Type* type,
      Expression* value, Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~FiberParameter();

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
