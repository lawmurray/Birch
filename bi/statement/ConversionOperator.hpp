/**
 * @file
 */
#pragma once

#include "bi/statement/Statement.hpp"
#include "bi/common/ReturnTyped.hpp"
#include "bi/common/Scoped.hpp"
#include "bi/common/Braced.hpp"

namespace bi {
/**
 * Type conversion operator.
 *
 * @ingroup birch_statement
 */
class ConversionOperator: public Statement,
    public ReturnTyped,
    public Scoped,
    public Braced {
public:
  /**
   * Constructor.
   *
   * @param returnType Return type.
   * @param braces Body.
   * @param loc Location.
   */
  ConversionOperator(Type* returnType, Statement* braces,
      Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~ConversionOperator();

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
