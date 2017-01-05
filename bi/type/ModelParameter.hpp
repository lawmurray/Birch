/**
 * @file
 */
#pragma once

#include "bi/type/Type.hpp"
#include "bi/common/Parenthesised.hpp"
#include "bi/common/Based.hpp"
#include "bi/common/Named.hpp"
#include "bi/common/Braced.hpp"
#include "bi/common/Scoped.hpp"
#include "bi/common/Parameter.hpp"

namespace bi {
/**
 * Type parameter.
 *
 * @ingroup compiler_type
 */
class ModelParameter: public Type,
    public Named,
    public Parenthesised,
    public Based,
    public Braced,
    public Scoped,
    public Parameter<Type> {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param parens Parentheses.
   * @param op Operator giving relation to base type.
   * @param base Base type.
   * @param braces Braces.
   * @param loc Location.
   */
  ModelParameter(shared_ptr<Name> name, Expression* parens, shared_ptr<Name> op,
      Type* base, Expression* braces, shared_ptr<Location> loc = nullptr);

  /**
   * Destructor.
   */
  virtual ~ModelParameter();

  /**
   * Get all member variables.
   */
  const std::list<VarParameter*>& vars() const;

  /**
   * Get all member functions.
   */
  const std::list<FuncParameter*>& funcs() const;

  virtual Type* accept(Cloner* visitor) const;
  virtual Type* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  bool builtin() const;

  /**
   * Constructor.
   */
  shared_ptr<FuncParameter> constructor;

  /**
   * Assignment operator.
   */
  shared_ptr<FuncParameter> assignment;

  virtual bool dispatch(Type& o);
  virtual bool le(ModelParameter& o);
  virtual bool le(EmptyType& o);
};
}
