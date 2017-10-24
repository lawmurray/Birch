/**
 * @file
 */
#pragma once

#include "bi/visitor/Modifier.hpp"
#include "bi/visitor/Cloner.hpp"
#include "bi/visitor/Assigner.hpp"

#include <stack>
#include <list>

namespace bi {
/**
 * Abstract class for resolvers, with common functionality.
 *
 * @ingroup compiler_visitor
 */
class Resolver: public Modifier {
public:
  /**
   * Constructor.
   */
  Resolver();

  /**
   * Destructor.
   */
  virtual ~Resolver() = 0;

  using Modifier::modify;

  virtual File* modify(File* file);

  virtual Expression* modify(List<Expression>* o);
  virtual Expression* modify(Parentheses* o);
  virtual Expression* modify(Binary* o);

  virtual Type* modify(TypeIdentifier* o);
  virtual Type* modify(ClassType* o);
  virtual Type* modify(AliasType* o);
  virtual Type* modify(BasicType* o);

protected:
  /**
   * Resolve an identifier.
   *
   * @tparam ObjectType Object type.
   *
   * @param o The identifier.
   */
  template<class ObjectType>
  void resolve(ObjectType* o);

  /**
   * Look up a reference that is syntactically ambiguous in an expression
   * context.
   *
   * @param ref The reference.
   * @param scope The membership scope, if it is to be used for lookup,
   * otherwise the containing scope is used.
   *
   * @return A new, unambiguous, reference.
   */
  Expression* lookup(Identifier<Unknown>* ref, Scope* scope = nullptr);

  /**
   * Look up a reference that is syntactically ambiguous in a type
   * context.
   *
   * @param ref The reference.
   * @param scope The membership scope, if it is to be used for lookup,
   * otherwise the containing scope is used.
   *
   * @return A new, unambiguous, reference.
   */
  Type* lookup(TypeIdentifier* ref, Scope* scope = nullptr);

  /**
   * Check that an expression is of boolean type.
   *
   * @param o The expression.
   */
  void checkBoolean(const Expression* o);

  /**
   * Check that an expression is of integer type.
   *
   * @param o The expression.
   */
  void checkInteger(const Expression* o);

  /**
   * List of scopes, innermost at the back.
   */
  std::list<Scope*> scopes;

  /**
   * Scope for resolution of type members.
   */
  Scope* memberScope;

  /**
   * Current class.
   */
  Class* currentClass;

  /*
   * Auxiliary visitors.
   */
  Cloner cloner;
  Assigner assigner;
};
}

#include "bi/exception/all.hpp"

template<class ObjectType>
void bi::Resolver::resolve(ObjectType* o) {
  if (memberScope) {
    /* use the scope for the current member lookup */
    memberScope->resolve(o);
    memberScope = nullptr;
  } else {
    /* use current stack of scopes */
    for (auto iter = scopes.rbegin(); !o->target && iter != scopes.rend();
        ++iter) {
      (*iter)->resolve(o);
    }
  }
  if (!o->target) {
    throw UnresolvedException(o);
  }
}
