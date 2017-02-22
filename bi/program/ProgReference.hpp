/**
 * @file
 */
#pragma once

#include "bi/program/Prog.hpp"
#include "bi/program/ProgParameter.hpp"
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
class ProgReference: public Prog,
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
      shared_ptr<Location> loc = nullptr, ProgParameter* target = nullptr);

  /**
   * Destructor.
   */
  virtual ~ProgReference();

  virtual Prog* accept(Cloner* visitor) const;
  virtual Prog* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  using Prog::definitely;
  using Prog::possibly;

  virtual bool dispatchDefinitely(Prog& o);
  virtual bool definitely(ProgParameter& o);
  virtual bool definitely(ProgReference& o);

  virtual bool dispatchPossibly(Prog& o);
  virtual bool possibly(ProgParameter& o);
  virtual bool possibly(ProgReference& o);
};
}
