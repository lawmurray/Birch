/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/expression/VarParameter.hpp"
#include "bi/common/Named.hpp"
#include "bi/common/Reference.hpp"

namespace bi {
/**
 * Reference to a variable.
 *
 * @ingroup compiler_expression
 */
class VarReference: public Expression, public Named, public Reference<
    VarParameter> {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param brackets Ranging/indexing expression in square brackets.
   * @param loc Location.
   * @param target Target.
   */
  VarReference(shared_ptr<Name> name, shared_ptr<Location> loc = nullptr,
      const VarParameter* target = nullptr);

  /**
   * Destructor.
   */
  virtual ~VarReference();

  virtual Expression* acceptClone(Cloner* visitor) const;
  virtual void acceptModify(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual bool operator<=(Expression& o);
  virtual bool operator==(const Expression& o) const;
};
}
