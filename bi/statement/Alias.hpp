/**
 * @file
 */
#pragma once

#include "bi/statement/Statement.hpp"
#include "bi/common/Named.hpp"
#include "bi/common/Numbered.hpp"
#include "bi/common/Based.hpp"

namespace bi {
/**
 * Alias of another type.
 *
 * @ingroup birch_statement
 */
class Alias: public Statement, public Named, public Numbered, public Based {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param base Base type.
   * @param loc Location.
   */
  Alias(Name* name, Type* base, Location* loc = nullptr);

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
};
}
