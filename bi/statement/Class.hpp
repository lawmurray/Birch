/**
 * @file
 */
#pragma once

#include "bi/statement/Statement.hpp"
#include "bi/common/Named.hpp"
#include "bi/common/Based.hpp"
#include "bi/common/Braced.hpp"
#include "bi/common/Scoped.hpp"

namespace bi {
/**
 * Class.
 *
 * @ingroup compiler_type
 */
class Class: public Statement,
    public Named,
    public Based,
    public Braced,
    public Scoped {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param base Base type.
   * @param braces Braces.
   * @param loc Location.
   */
  Class(shared_ptr<Name> name, Type* base, Statement* braces,
      shared_ptr<Location> loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~Class();

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  /**
   * Iterators over type conversions.
   */
  auto beginConversions() const {
    return scope->conversionOperators.params.begin();
  }
  auto endConversions() const {
    return scope->conversionOperators.params.end();
  }

  using Statement::definitely;
  using Statement::possibly;

  virtual bool dispatchDefinitely(const Statement& o) const;
  virtual bool definitely(const Class& o) const;

  virtual bool dispatchPossibly(const Statement& o) const;
  virtual bool possibly(const Class& o) const;
};
}
