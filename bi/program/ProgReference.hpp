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
      shared_ptr<Location> loc = nullptr, const ProgParameter* target = nullptr);

  /**
   * Destructor.
   */
  virtual ~ProgReference();

  virtual Prog* acceptClone(Cloner* visitor) const;
  virtual Prog* acceptModify(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual bool operator<=(Prog& o);
  virtual bool operator==(const Prog& o) const;
};
}

inline bi::ProgReference::~ProgReference() {
  //
}
