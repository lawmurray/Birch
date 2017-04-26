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

namespace bi {
/**
 * Type form.
 */
enum TypeForm {
  STRUCT_TYPE,
  CLASS_TYPE,
  BUILTIN_TYPE,
  ALIAS_TYPE
};

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
    public Scoped {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param parens Parentheses.
   * @param base Base type.
   * @param braces Braces.
   * @param form Type form.
   * @param loc Location.
   * @param assignable Is this type writeable?
   */
  ModelParameter(shared_ptr<Name> name, Expression* parens,
      Type* base, Expression* braces, const TypeForm form,
      shared_ptr<Location> loc = nullptr, const bool assignable = false);

  /**
   * Destructor.
   */
  virtual ~ModelParameter();

  virtual Type* accept(Cloner* visitor) const;
  virtual Type* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  bool isBuiltin() const;
  bool isStruct() const;
  bool isClass() const;
  bool isAlias() const;

  /**
   * Get the base type.
   */
  const ModelParameter* getBase() const;

  /**
   * Is this type equal to or less than @p o by inheritance?
   */
  bool canUpcast(const ModelParameter* o) const;

  /**
   * Is this type equal to or greater than @p o by inheritance?
   */
  bool canDowncast(const ModelParameter* o) const;

  using Type::definitely;
  using Type::possibly;

  virtual bool dispatchDefinitely(const Type& o) const;
  virtual bool definitely(const ModelParameter& o) const;
  virtual bool definitely(const ParenthesesType& o) const;

  virtual bool dispatchPossibly(const Type& o) const;
  virtual bool possibly(const ModelParameter& o) const;
  virtual bool possibly(const ParenthesesType& o) const;

  /**
   * Type form.
   */
  TypeForm form;
};
}
