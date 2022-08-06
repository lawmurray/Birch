/**
 * @file
 */
#pragma once

#include "src/statement/Statement.hpp"
#include "src/common/Annotated.hpp"
#include "src/common/ReturnTyped.hpp"
#include "src/common/Braced.hpp"

namespace birch {
/**
 * Type conversion operator.
 *
 * @ingroup statement
 */
class ConversionOperator: public Statement,
    public Annotated,
    public ReturnTyped,
    public Braced {
public:
  /**
   * Constructor.
   *
   * @param annotation Annotation.
   * @param returnType Return type.
   * @param braces Body.
   * @param loc Location.
   */
  ConversionOperator(const Annotation annotation, Type* returnType,
      Statement* braces, Location* loc = nullptr);

  virtual void accept(Visitor* visitor) const;
};
}
