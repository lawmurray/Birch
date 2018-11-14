/**
 * @file
 */
#pragma once

#include "bi/statement/Statement.hpp"
#include "bi/common/Single.hpp"

namespace bi {
/**
 * Statement indicating that a particular instantiation of a generic object
 * has been compiled by a dependency, and should not be compiled again.
 *
 * @ingroup statement
 */
template<class T>
class Instantiated: public Statement, public Single<T> {
public:
  /**
   * Constructor.
   *
   * @param single Class type.
   * @param loc Location.
   */
  Instantiated(T* single, Location* loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~Instantiated();

  virtual Statement* accept(Cloner* visitor) const;
  virtual Statement* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;
};
}
