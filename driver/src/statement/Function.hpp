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
#include "src/common/ReturnTyped.hpp"
#include "src/common/Braced.hpp"
#include "src/common/Scoped.hpp"

namespace birch {
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

  virtual bool isMember() const;

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
