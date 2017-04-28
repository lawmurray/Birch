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
class TypeParameter: public Type,
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
  TypeParameter(shared_ptr<Name> name, Expression* parens,
      Type* base, Expression* braces, const TypeForm form,
      shared_ptr<Location> loc = nullptr, const bool assignable = false);

  /**
   * Destructor.
   */
  virtual ~TypeParameter();

  virtual Type* accept(Cloner* visitor) const;
  virtual Type* accept(Modifier* visitor);
  virtual void accept(Visitor* visitor) const;

  /**
   * Get the base type.
   */
  const TypeParameter* super() const;

  /**
   * Return the canonical representation of this type. For an alias, this is
   * the base type, for all other types, the same type.
   */
  const TypeParameter* canonical() const;

  /**
   * Iterators over type conversions.
   */
  auto beginConversions() const {
    return scope->convs.params.begin();
  }
  auto endConversions() const {
    return scope->convs.params.end();
  }

  bool isBuiltin() const;
  bool isStruct() const;
  bool isClass() const;
  bool isAlias() const;

  using Type::definitely;
  using Type::possibly;

  virtual bool dispatchDefinitely(const Type& o) const;
  virtual bool definitely(const TypeParameter& o) const;
  virtual bool definitely(const ParenthesesType& o) const;

  virtual bool dispatchPossibly(const Type& o) const;
  virtual bool possibly(const TypeParameter& o) const;
  virtual bool possibly(const ParenthesesType& o) const;

  /**
   * Type form.
   */
  TypeForm form;
};
}
