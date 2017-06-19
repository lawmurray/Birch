/**
 * @file
 */
#pragma once

#include "bi/statement/Statement.hpp"
#include "bi/common/Named.hpp"
#include "bi/common/Based.hpp"

namespace bi {
/**
 * Alias of another type.
 *
 * @ingroup compiler_type
 */
class Alias: public Statement, public Named, public Based {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param base Base type.
   * @param loc Location.
   */
  Alias(shared_ptr<Name> name, Type* base,
      shared_ptr<Location> loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~Alias();

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  /**
   * Base class.
   */
  const Statement* super() const;

  /**
   * Base class.
   */
  const Statement* canonical() const;

  using Statement::definitely;
  using Statement::possibly;

  virtual bool dispatchDefinitely(const Statement& o) const;
  virtual bool definitely(const Alias& o) const;

  virtual bool dispatchPossibly(const Statement& o) const;
  virtual bool possibly(const Alias& o) const;
};
}
