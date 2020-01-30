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
#include "bi/common/ReturnTyped.hpp"
#include "bi/common/Braced.hpp"
#include "bi/common/Scoped.hpp"

namespace bi {
/**
 * Function.
 *
 * @ingroup statement
 */
class Function: public Statement,
    public Annotated,
    public Named,
    public Numbered,
    public TypeParameterised,
    public Parameterised,
    public ReturnTyped,
    public Scoped,
    public Braced {
public:
  /**
   * Constructor.
   *
   * @param annotation Annotation.
   * @param name Name.
   * @param typeParams Generic type parameters.
   * @param params Parameters.
   * @param returnType Return type.
   * @param braces Body.
   * @param loc Location.
   */
  Function(const Annotation annotation, Name* name, Expression* typeParams,
      Expression* params, Type* returnType, Statement* braces, Location* loc =
          nullptr);

  /**
   * Destructor.
   */
  virtual ~Function();

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
