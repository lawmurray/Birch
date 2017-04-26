/**
 * @file
 */
#pragma once

#include "bi/type/Type.hpp"

#include "bi/primitive/unique_ptr.hpp"

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
   * @param definite Definite type.
   * @param possibles Possible types.
   * @param loc Location.
   * @param assignable Is this type assignable?
   */
  VariantType(Type* definite,
      const std::list<Type*>& possibles = std::list<Type*>(),
      shared_ptr<Location> loc = nullptr, const bool assignable = false);

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

  virtual bool isVariant() const;

  using Type::definitely;
  using Type::possibly;

  virtual bool possiblyAny(const Type& o) const;

  virtual bool dispatchDefinitely(const Type& o) const;
  virtual bool definitely(const BracketsType& o) const;
  virtual bool definitely(const EmptyType& o) const;
  virtual bool definitely(const LambdaType& o) const;
  virtual bool definitely(const List<Type>& o) const;
  virtual bool definitely(const ModelParameter& o) const;
  virtual bool definitely(const ModelReference& o) const;
  virtual bool definitely(const ParenthesesType& o) const;
  virtual bool definitely(const DelayType& o) const;

  virtual bool dispatchPossibly(const Type& o) const;
  virtual bool possibly(const BracketsType& o) const;
  virtual bool possibly(const EmptyType& o) const;
  virtual bool possibly(const LambdaType& o) const;
  virtual bool possibly(const List<Type>& o) const;
  virtual bool possibly(const ModelParameter& o) const;
  virtual bool possibly(const ModelReference& o) const;
  virtual bool possibly(const ParenthesesType& o) const;
  virtual bool possibly(const DelayType& o) const;

  /**
   * Definite type.
   */
  unique_ptr<Type> definite;

  /**
   * Possible types.
   */
  std::list<Type*> possibles;
};
}
