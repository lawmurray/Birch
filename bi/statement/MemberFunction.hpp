/**
 * @file
 */
#pragma once

#include "bi/statement/Statement.hpp"
#include "bi/common/Named.hpp"
#include "bi/common/Numbered.hpp"
#include "bi/common/Parenthesised.hpp"
#include "bi/common/ReturnTyped.hpp"
#include "bi/common/Braced.hpp"
#include "bi/common/Scoped.hpp"

namespace bi {
/**
 * Class member function.
 *
 * @ingroup compiler_expression
 */
class MemberFunction: public Statement,
    public Named,
    public Numbered,
    public Parenthesised,
    public ReturnTyped,
    public Scoped,
    public Braced {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param parens Parentheses expression.
   * @param returnType Return type.
   * @param braces Braces expression.
   * @param loc Location.
   */
  MemberFunction(shared_ptr<Name> name, Expression* parens, Type* returnType,
      Statement* braces, shared_ptr<Location> loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~MemberFunction();

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  using Statement::definitely;
  using Statement::possibly;

  virtual bool dispatchDefinitely(const Statement& o) const;
  virtual bool definitely(const MemberFunction& o) const;

  virtual bool dispatchPossibly(const Statement& o) const;
  virtual bool possibly(const MemberFunction& o) const;
};
}
