/**
 * @file
 */
#pragma once

#include "bi/statement/Statement.hpp"
#include "bi/common/Annotated.hpp"
#include "bi/common/Named.hpp"
#include "bi/common/Numbered.hpp"
#include "bi/common/TypeParameterised.hpp"
#include "bi/common/Parameterised.hpp"
#include "bi/common/Based.hpp"
#include "bi/common/Argumented.hpp"
#include "bi/common/Braced.hpp"
#include "bi/common/Scoped.hpp"

namespace bi {
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

  /**
   * Destructor.
   */
  virtual ~Class();

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  /**
   * Scope for initialization parameters.
   */
  Scope* initScope;
};
}
