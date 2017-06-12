/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/statement/Function.hpp"
#include "bi/common/Named.hpp"
#include "bi/common/Parenthesised.hpp"
#include "bi/common/Reference.hpp"

#include <list>

namespace bi {
/**
 * Reference to function.
 *
 * @ingroup compiler_expression
 */
class FuncReference: public Expression,
    public Named,
    public Parenthesised,
    public Reference<Function> {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param parens Expression in parentheses.
   * @param loc Location.
   * @param target Target.
   */
  FuncReference(shared_ptr<Name> name, Expression* parens,
      shared_ptr<Location> loc = nullptr, const Function* target =
          nullptr);

  /**
   * Destructor.
   */
  virtual ~FuncReference();

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  using Expression::definitely;
  using Expression::possibly;

  virtual bool dispatchDefinitely(const Expression& o) const;
  virtual bool definitely(const FuncReference& o) const;
  virtual bool definitely(const Function& o) const;
  virtual bool definitely(const VarParameter& o) const;

  virtual bool dispatchPossibly(const Expression& o) const;
  virtual bool possibly(const FuncReference& o) const;
  virtual bool possibly(const Function& o) const;
  virtual bool possibly(const VarParameter& o) const;
};
}
