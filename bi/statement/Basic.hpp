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
 * Basic (built-in) type.
 *
 * @ingroup compiler_statement
 */
class Basic: public Statement, public Named, public Numbered, public Based {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param base Base type.
   * @param loc Location.
   */
  Basic(Name* name, Type* base, Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~Basic();

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
