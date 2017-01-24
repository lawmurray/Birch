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
      new EmptyExpression(),
      shared_ptr<Location> loc = nullptr, ModelParameter* target = nullptr);

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

  virtual bool builtin() const;

  /**
   * Does this model inherit from another?
   */
  virtual possibly isa(ModelReference& o);

  virtual Type* accept(Cloner* visitor) const;
  virtual Type* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual possibly dispatch(Type& o);
  virtual possibly le(ModelParameter& o);
  virtual possibly le(ModelReference& o);
  virtual possibly le(EmptyType& o);
};
}
