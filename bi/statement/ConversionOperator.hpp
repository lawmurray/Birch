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
 * @ingroup compiler_statement
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
   * @param braces Braces expression.
   * @param loc Location.
   */
  ConversionOperator(Type* returnType, Statement* braces,
      shared_ptr<Location> loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~ConversionOperator();

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  using Statement::definitely;
  using Statement::possibly;

  virtual bool dispatchDefinitely(const Statement& o) const;
  virtual bool definitely(const ConversionOperator& o) const;

  virtual bool dispatchPossibly(const Statement& o) const;
  virtual bool possibly(const ConversionOperator& o) const;
};
}
