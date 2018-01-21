/**
 * @file
 */
#pragma once

#include "bi/statement/Statement.hpp"
#include "bi/common/Named.hpp"
#include "bi/common/Numbered.hpp"
#include "bi/common/Based.hpp"

#include <set>

namespace bi {
/**
 * Basic type.
 *
 * @ingroup statement
 */
class Basic: public Statement, public Named, public Numbered, public Based {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param base Base type.
   * @param alias Is this an alias relationship?
   * @param loc Location.
   */
  Basic(Name* name, Type* base, const bool alias, Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~Basic();

  /**
   * Add a super type.
   */
  virtual void addSuper(const Type* o);

  /**
   * Is the given type a super type of this?
   */
  virtual bool hasSuper(const Type* o) const;

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

private:
  /**
   * Super classes.
   */
  std::set<const Basic*> supers;
};
}
