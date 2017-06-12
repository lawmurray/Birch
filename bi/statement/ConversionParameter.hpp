/**
 * @file
 */
#pragma once

#include "bi/statement/Statement.hpp"
#include "bi/expression/Expression.hpp"
#include "bi/common/Braced.hpp"
#include "bi/common/Signature.hpp"
#include "bi/common/Typed.hpp"
#include "bi/common/Scoped.hpp"

namespace bi {
/**
 * Type conversion function.
 *
 * @ingroup compiler_expression
 */
class ConversionParameter: public Statement,
    public Typed,
    public Scoped,
    public Braced {
public:
  /**
   * Constructor.
   *
   * @param type Typed type.
   * @param braces Braces expression.
   * @param loc Location.
   */
  ConversionParameter(Type* type, Expression* braces,
      shared_ptr<Location> loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~ConversionParameter();

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  using Statement::definitely;
  using Statement::possibly;

  virtual bool dispatchDefinitely(const Statement& o) const;
  virtual bool definitely(const ConversionParameter& o) const;

  virtual bool dispatchPossibly(const Statement& o) const;
  virtual bool possibly(const ConversionParameter& o) const;
};
}
