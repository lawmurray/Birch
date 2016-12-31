/**
 * @file
 */
#pragma once

#include "bi/type/Type.hpp"
#include "bi/type/ModelParameter.hpp"
#include "bi/expression/EmptyExpression.hpp"
#include "bi/common/Bracketed.hpp"
#include "bi/common/Reference.hpp"

namespace bi {
/**
 * Reference to model.
 *
 * @ingroup compiler_type
 */
class ModelReference: public Type,
    public Named,
    public Bracketed,
    public Reference<ModelParameter> {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param brackets Square brackets.
   * @param loc Location.
   * @param target Target.
   */
  ModelReference(shared_ptr<Name> name, Expression* brackets =
      new EmptyExpression(), shared_ptr<Location> loc = nullptr,
      const ModelParameter* target = nullptr);

  /**
   * Constructor.
   *
   * @param name Name.
   * @param ndims Number of dimensions.
   * @param target Target.
   */
  ModelReference(shared_ptr<Name> name, const int ndims,
      const ModelParameter* target = nullptr);

  /**
   * Destructor.
   */
  virtual ~ModelReference();

  virtual bool builtin() const;
  virtual int count() const;

  virtual Type* acceptClone(Cloner* visitor) const;
  virtual Type* acceptModify(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  virtual bool operator<=(Type& o);
  virtual bool operator==(const Type& o) const;

  /**
   * Number of dimensions.
   */
  int ndims;
};
}
