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
class Basic: public Statement, public Named {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param loc Location.
   */
  Basic(shared_ptr<Name> name, shared_ptr<Location> loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~Basic();

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  using Statement::definitely;
  using Statement::possibly;

  virtual bool dispatchDefinitely(const Statement& o) const;
  virtual bool definitely(const Basic& o) const;

  virtual bool dispatchPossibly(const Statement& o) const;
  virtual bool possibly(const Basic& o) const;
};
}
