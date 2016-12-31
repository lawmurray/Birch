/**
 * @file
 */
#pragma once

#include "bi/expression/Expression.hpp"
#include "bi/expression/FuncParameter.hpp"
#include "bi/common/Named.hpp"
#include "bi/common/Reference.hpp"
#include "bi/common/Parenthesised.hpp"
#include "bi/common/Formed.hpp"

#include <list>

namespace bi {
/**
 * Reference to function.
 *
 * @ingroup compiler_expression
 */
class FuncReference: public Expression, public Named, public Reference<
    FuncParameter>, public Parenthesised, public Formed {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param parens Expression in parentheses.
   * @param form Function form.
   * @param loc Location.
   * @param target Target.
   */
  FuncReference(shared_ptr<Name> name, Expression* parens, const FunctionForm form,
      shared_ptr<Location> loc = nullptr, const FuncParameter* target =
          nullptr);

  /**
   * Destructor.
   */
  virtual ~FuncReference();

  virtual Expression* acceptClone(Cloner* visitor) const;
  virtual Expression* acceptModify(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual bool operator<=(Expression& o);
  virtual bool operator==(const Expression& o) const;

  /**
   * Arguments.
   */
  std::list<const Expression*> args;
};
}
