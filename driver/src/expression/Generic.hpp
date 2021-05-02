/**
 * @file
 */
#pragma once

#include "src/expression/Expression.hpp"
#include "src/common/Annotated.hpp"
#include "src/common/Named.hpp"
#include "src/common/Numbered.hpp"
#include "src/common/Typed.hpp"

namespace birch {
/**
 * Generic type.
 *
 * @ingroup expression
 */
class Generic: public Expression,
    public Annotated,
    public Named,
    public Numbered,
    public Typed {
public:
  /**
   * Constructor.
   *
   * @param annotation Annotation.
   * @param name Name.
   * @param type Type.
   * @param loc Location.
   */
  Generic(const Annotation annotation, Name* name, Type* type,
      Location* loc = nullptr);

  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
