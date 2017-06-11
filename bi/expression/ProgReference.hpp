/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/expression/ProgParameter.hpp"
#include "bi/expression/EmptyExpression.hpp"
#include "bi/common/Named.hpp"
#include "bi/common/Parenthesised.hpp"
#include "bi/common/Reference.hpp"

#include <list>

namespace bi {
/**
 * Reference to function.
 *
 * @ingroup compiler_program
 */
class ProgReference: public Expression,
    public Named,
    public Parenthesised,
    public Reference<ProgParameter> {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param parens Expression in parentheses.
   * @param loc Location.
   */
  ProgReference(shared_ptr<Name> name, Expression* parens,
      shared_ptr<Location> loc = nullptr, const ProgParameter* target =
          nullptr);

  /**
   * Destructor.
   */
  virtual ~ProgReference();

  virtual Expression* accept(Cloner* visitor) const;
  virtual Expression* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  using Expression::definitely;
  using Expression::possibly;

  virtual bool dispatchDefinitely(const Expression& o) const;
  virtual bool definitely(const ProgParameter& o) const;
  virtual bool definitely(const ProgReference& o) const;

  virtual bool dispatchPossibly(const Expression& o) const;
  virtual bool possibly(const ProgParameter& o) const;
  virtual bool possibly(const ProgReference& o) const;
};
}
