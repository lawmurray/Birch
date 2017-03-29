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
   * @param definite Definite type.
   * @param possibles Possible types.
   * @param loc Location.
   * @param assignable Is this type writeable?
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

  virtual bool definitelyAll(Type& o);
  virtual bool possiblyAny(Type& o);

  virtual bool dispatchDefinitely(Type& o);
  virtual bool definitely(AssignableType& o);
  virtual bool definitely(BracketsType& o);
  virtual bool definitely(EmptyType& o);
  virtual bool definitely(LambdaType& o);
  virtual bool definitely(List<Type>& o);
  virtual bool definitely(ModelParameter& o);
  virtual bool definitely(ModelReference& o);
  virtual bool definitely(ParenthesesType& o);
  virtual bool definitely(RandomType& o);
  virtual bool definitely(VariantType& o);

  virtual bool dispatchPossibly(Type& o);
  virtual bool possibly(AssignableType& o);
  virtual bool possibly(BracketsType& o);
  virtual bool possibly(EmptyType& o);
  virtual bool possibly(LambdaType& o);
  virtual bool possibly(List<Type>& o);
  virtual bool possibly(ModelParameter& o);
  virtual bool possibly(ModelReference& o);
  virtual bool possibly(ParenthesesType& o);
  virtual bool possibly(RandomType& o);
  virtual bool possibly(VariantType& o);

  /**
   * Definite type.
   */
  Type* definite;

  /**
   * Possible types.
   */
  std::list<Type*> possibles;
};
}
