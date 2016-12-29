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

  virtual Prog* acceptClone(Cloner* visitor) const;
  virtual void acceptModify(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual bool operator<=(Prog& o);
  virtual bool operator==(const Prog& o) const;

  /**
   * Input parameters
   */
  std::list<const VarParameter*> inputs;
};
}

inline bi::ProgParameter::~ProgParameter() {
  //
}
