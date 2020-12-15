/**
 * @file
 */
#pragma once

#include "src/statement/Statement.hpp"
#include "src/common/Numbered.hpp"
#include "src/common/Parameterised.hpp"
#include "src/common/ReturnTyped.hpp"
#include "src/common/Scoped.hpp"
#include "src/common/Braced.hpp"

namespace birch {
/**
 * Slice operator.
 *
 * @ingroup statement
 */
class SliceOperator:
    public Statement,
    public Numbered,
    public Parameterised,
    public ReturnTyped,
    public Scoped,
    public Braced {
public:
  /**
   * Constructor.
   *
   * @param params Parameters.
   * @param braces Body.
   * @param loc Location.
   */
  SliceOperator(Expression* params, Type* returnType, Statement* braces,
      Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~SliceOperator();

  virtual bool isDeclaration() const;

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
