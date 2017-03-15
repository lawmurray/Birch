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
   * Possible types.
   */
  std::list<Type*> types;
};
}
