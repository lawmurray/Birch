/**
 * @file
 */
#pragma once

#include "bi/statement/Statement.hpp"
#include "bi/common/Named.hpp"
#include "bi/common/Numbered.hpp"

namespace bi {
/**
 * Generic type.
 *
 * @ingroup compiler_statement
 */
class Generic: public Statement, public Named, public Numbered {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param loc Location.
   */
  Generic(Name* name, Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~Generic();

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
