/**
 * @file
 */
#pragma once

#include "bi/program/Prog.hpp"
#include "bi/common/Named.hpp"
#include "bi/common/Scoped.hpp"
#include "bi/common/Parameter.hpp"
#include "bi/common/Parenthesised.hpp"
#include "bi/common/Braced.hpp"

namespace bi {
/**
 * Program parameter.
 *
 * @ingroup compiler_program
 */
class ProgParameter: public Prog,
    public Named,
    public Scoped,
    public Parenthesised,
    public Braced,
    public Parameter<Prog> {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param parens Expression in parentheses.
   * @param braces Expression in braces.
   * @param loc Location.
   */
  ProgParameter(shared_ptr<Name> name, Expression* parens, Expression* braces,
      shared_ptr<Location> loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~ProgParameter();

  virtual Prog* accept(Cloner* visitor) const;
  virtual Prog* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  using Prog::definitely;
  using Prog::possibly;

  virtual bool dispatchDefinitely(Prog& o);
  virtual bool definitely(ProgParameter& o);

  virtual bool dispatchPossibly(Prog& o);
  virtual bool possibly(ProgParameter& o);
};
}
