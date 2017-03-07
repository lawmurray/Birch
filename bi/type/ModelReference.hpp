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
   * @param target Target.
   */
  ModelReference(shared_ptr<Name> name, Expression* parens =
      new EmptyExpression(), shared_ptr<Location> loc = nullptr,
      ModelParameter* target = nullptr);

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

  /**
   * Is this type equal to or less than @p o by inheritance?
   */
  bool canUpcast(ModelReference& o);

  /**
   * Is this type greater than @p o by inheritance?
   */
  bool canDowncast(ModelReference& o);

  virtual Type* accept(Cloner* visitor) const;
  virtual Type* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  using Type::definitely;
  using Type::possibly;

  virtual bool dispatchDefinitely(Type& o);
  virtual bool definitely(ModelParameter& o);
  virtual bool definitely(ModelReference& o);
  virtual bool definitely(EmptyType& o);

  virtual bool dispatchPossibly(Type& o);
  virtual bool possibly(ModelParameter& o);
  virtual bool possibly(ModelReference& o);
  virtual bool possibly(EmptyType& o);
};
}
