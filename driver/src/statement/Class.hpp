/**
 * @file
 */
#pragma once

#include "src/statement/Statement.hpp"
#include "src/common/Annotated.hpp"
#include "src/common/Named.hpp"
#include "src/common/Numbered.hpp"
#include "src/common/TypeParameterised.hpp"
#include "src/common/Parameterised.hpp"
#include "src/common/Based.hpp"
#include "src/common/Argumented.hpp"
#include "src/common/Braced.hpp"
#include "src/common/Scoped.hpp"

namespace birch {
/**
 * Class.
 *
 * @ingroup statement
 */
class Class: public Statement,
    public Annotated,
    public Named,
    public Numbered,
    public TypeParameterised,
    public Parameterised,
    public Based,
    public Argumented,
    public Scoped,
    public Braced {
public:
  /**
   * Constructor.
   *
   * @param annotation Annotation.
   * @param name Name.
   * @param typeParams Generic type parameters.
   * @param params Constructor parameters.
   * @param base Base type.
   * @param alias Is this an alias relationship?
   * @param args Base type constructor arguments.
   * @param braces Braces.
   * @param loc Location.
   */
  Class(const Annotation annotation, Name* name, Expression* typeParams,
      Expression* params, Type* base, const bool alias, Expression* args,
      Statement* braces, Location* loc = nullptr);

  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  /**
   * Scope for initialization parameters.
   */
  Scope* initScope;
};
}
