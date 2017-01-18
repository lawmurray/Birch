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

  virtual possibly dispatch(Prog& o);
  virtual possibly le(ProgParameter& o);
  virtual possibly le(ProgReference& o);
};
}
