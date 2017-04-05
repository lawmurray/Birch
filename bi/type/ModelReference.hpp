/**
 * @file
 */
#pragma once

#include "bi/type/Type.hpp"
#include "bi/type/ModelParameter.hpp"
#include "bi/expression/EmptyExpression.hpp"
#include "bi/common/Parenthesised.hpp"
#include "bi/common/Reference.hpp"
#include "bi/type/EmptyType.hpp"

namespace bi {
/**
 * Reference to model.
 *
 * @ingroup compiler_type
 */
class ModelReference: public Type,
    public Named,
    public Parenthesised,
    public Reference<ModelParameter> {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param parens Parentheses.
   * @param loc Location.
   * @param assignable Is this type assignable?
   * @param polymorphic Is this type polymorphic?
   * @param target Target.
   */
  ModelReference(shared_ptr<Name> name, Expression* parens =
      new EmptyExpression(), shared_ptr<Location> loc = nullptr,
      const bool assignable = false,
      const bool polymorphic = false, ModelParameter* target = nullptr);

  /**
   * Constructor.
   *
   * @param target Target.
   */
  ModelReference(ModelParameter* target);

  /**
   * Destructor.
   */
  virtual ~ModelReference();

  virtual bool isBuiltin() const;
  virtual bool isModel() const;

  virtual Type* accept(Cloner* visitor) const;
  virtual Type* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  using Type::definitely;
  using Type::possibly;

  virtual bool dispatchDefinitely(const Type& o) const;
  virtual bool definitely(const ModelParameter& o) const;
  virtual bool definitely(const ModelReference& o) const;
  virtual bool definitely(const EmptyType& o) const;

  virtual bool dispatchPossibly(const Type& o) const;
  virtual bool possibly(const ModelParameter& o) const;
  virtual bool possibly(const ModelReference& o) const;
  virtual bool possibly(const EmptyType& o) const;
};
}
