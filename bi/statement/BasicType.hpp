/**
 * @file
 */
#pragma once

#include "bi/statement/Statement.hpp"
#include "bi/common/Named.hpp"

namespace bi {
/**
 * Basic (built-in) type.
 *
 * @ingroup compiler_type
 */
class BasicType: public Statement, public Named {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param loc Location.
   */
  BasicType(shared_ptr<Name> name, shared_ptr<Location> loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~BasicType();

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  using Statement::definitely;
  using Statement::possibly;

  virtual bool dispatchDefinitely(const Statement& o) const;
  virtual bool definitely(const BasicType& o) const;

  virtual bool dispatchPossibly(const Statement& o) const;
  virtual bool possibly(const BasicType& o) const;
};
}
