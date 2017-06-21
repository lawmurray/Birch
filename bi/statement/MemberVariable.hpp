/**
 * @file
 */
#pragma once

#include "bi/statement/Statement.hpp"
#include "bi/common/Named.hpp"
#include "bi/common/Numbered.hpp"
#include "bi/common/Typed.hpp"
#include "bi/primitive/unique_ptr.hpp"

namespace bi {
/**
 * Class member variable.
 *
 * @ingroup compiler_statement
 */
class MemberVariable: public Statement,
    public Named,
    public Numbered,
    public Typed {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param type Type.
   * @param loc Location.
   */
  MemberVariable(shared_ptr<Name> name, Type* type, shared_ptr<Location> loc =
      nullptr);

  /**
   * Destructor.
   */
  virtual ~MemberVariable();

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  using Statement::definitely;
  using Statement::possibly;

  virtual bool dispatchDefinitely(const Statement& o) const;
  virtual bool definitely(const MemberVariable& o) const;

  virtual bool dispatchPossibly(const Statement& o) const;
  virtual bool possibly(const MemberVariable& o) const;
};
}
