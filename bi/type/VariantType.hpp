/**
 * @file
 */
#pragma once

#include "bi/type/Type.hpp"

#include <list>

namespace bi {
/**
 * Variate type.
 *
 * @ingroup compiler_type
 */
class VariantType: public Type {
public:
  /**
   * Constructor.
   *
   * @param types Types.
   * @param loc Location.
   */
  VariantType(const std::list<Type*>& types, shared_ptr<Location> loc =
      nullptr);

  /**
   * Constructor.
   *
   * @param loc Location.
   */
  VariantType(shared_ptr<Location> loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~VariantType();

  /**
   * Add a new possible type.
   */
  void add(Type* o);

  /**
   * Total number of possible types, including definite type.
   */
  int size() const;

  virtual Type* accept(Cloner* visitor) const;
  virtual Type* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  using Type::definitely;
  using Type::possibly;

  virtual bool definitely(Type& o);
  virtual bool dispatchDefinitely(Type& o);

  virtual bool possibly(Type& o);
  virtual bool dispatchPossibly(Type& o);

  /**
   * Possible types.
   */
  std::list<Type*> types;
};
}
