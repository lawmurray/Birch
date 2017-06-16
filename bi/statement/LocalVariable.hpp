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
 * Global variable.
 *
 * @ingroup compiler_expression
 */
class LocalVariable: public Statement,
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
  LocalVariable(shared_ptr<Name> name, Type* type, shared_ptr<Location> loc =
      nullptr);

  /**
   * Destructor.
   */
  virtual ~LocalVariable();

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  using Statement::definitely;
  using Statement::possibly;

  virtual bool dispatchDefinitely(const Statement& o) const;
  virtual bool definitely(const LocalVariable& o) const;

  virtual bool dispatchPossibly(const Statement& o) const;
  virtual bool possibly(const LocalVariable& o) const;
};
}
