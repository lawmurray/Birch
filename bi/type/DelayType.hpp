/**
 * @file
 */
#pragma once

#include "bi/type/Type.hpp"
#include "bi/common/Binary.hpp"

namespace bi {
/**
 * Delay variate type.
 *
 * @ingroup compiler_type
 */
class DelayType: public Type, public TypeBinary {
public:
  /**
   * Constructor.
   *
   * @param left Variate type.
   * @param right Model type.
   * @param loc Location.
   * @param assignable Is this type assignable?
   */
  DelayType(Type* left, Type* right, shared_ptr<Location> loc = nullptr,
      const bool assignable = false);

  /**
   * Destructor.
   */
  virtual ~DelayType();

  virtual Type* accept(Cloner* visitor) const;
  virtual Type* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual bool isDelay() const;

  using Type::definitely;
  using Type::possibly;

  virtual bool dispatchDefinitely(const Type& o) const;
  virtual bool definitely(const DelayType& o) const;
  virtual bool definitely(const LambdaType& o) const;
  virtual bool definitely(const List<Type>& o) const;
  virtual bool definitely(const TypeParameter& o) const;
  virtual bool definitely(const TypeReference& o) const;
  virtual bool definitely(const ParenthesesType& o) const;

  virtual bool dispatchPossibly(const Type& o) const;
  virtual bool possibly(const DelayType& o) const;
  virtual bool possibly(const LambdaType& o) const;
  virtual bool possibly(const List<Type>& o) const;
  virtual bool possibly(const TypeParameter& o) const;
  virtual bool possibly(const TypeReference& o) const;
  virtual bool possibly(const ParenthesesType& o) const;
};
}
