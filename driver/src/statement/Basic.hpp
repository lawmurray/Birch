/**
 * @file
 */
#pragma once

#include "src/statement/Statement.hpp"
#include "src/common/Annotated.hpp"
#include "src/common/Named.hpp"
#include "src/common/Numbered.hpp"
#include "src/common/TypeParameterised.hpp"
#include "src/common/Based.hpp"

namespace birch {
/**
 * Basic type.
 *
 * @ingroup statement
 */
class Basic: public Statement,
    public Annotated,
    public Named,
    public Numbered,
    public TypeParameterised,
    public Based {
public:
  /**
   * Constructor.
   *
   * @param annotation Annotation.
   * @param name Name.
   * @param base Base type.
   * @param alias Is this an alias relationship?
   * @param loc Location.
   */
  Basic(const Annotation annotation, Name* name, Expression* typeParams,
      Type* base, const bool alias, Location* loc = nullptr);

  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
