/**
 * @file
 */
#pragma once

#include "src/statement/Statement.hpp"
#include "src/common/Annotated.hpp"
#include "src/common/Numbered.hpp"
#include "src/common/Parameterised.hpp"
#include "src/common/ReturnTyped.hpp"
#include "src/common/Braced.hpp"

namespace birch {
/**
 * Slice operator.
 *
 * @ingroup statement
 */
class SliceOperator: public Statement,
    public Annotated,
    public Numbered,
    public Parameterised,
    public ReturnTyped,
    public Braced {
public:
  /**
   * Constructor.
   *
   * @param annotation Annotation.
   * @param params Parameters.
   * @param braces Body.
   * @param loc Location.
   */
  SliceOperator(const Annotation annotation, Expression* params,
      Type* returnType, Statement* braces, Location* loc = nullptr);

  virtual void accept(Visitor* visitor) const;
};
}
