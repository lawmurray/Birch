/**
 * @file
 */
#pragma once

#include "bi/visitor/Modifier.hpp"
#include "bi/visitor/Cloner.hpp"

namespace bi {
/**
 * Abstract class for resolvers, with common functionality.
 *
 * @ingroup visitor
 */
class Resolver: public Modifier {
public:
  /**
   * Constructor.
   *
   * @param rootScope The root scope.
   * @param pointers Wrap class types in pointers?
   */
  Resolver(Scope* rootScope, const bool pointers);

  /**
   * Destructor.
   */
  virtual ~Resolver() = 0;

  using Modifier::modify;

  virtual Expression* modify(ExpressionList* o);
  virtual Expression* modify(Parentheses* o);
  virtual Expression* modify(Sequence* o);
  virtual Expression* modify(Binary* o);

  virtual Type* modify(UnknownType* o);
  virtual Type* modify(ClassType* o);
  virtual Type* modify(BasicType* o);
  virtual Type* modify(GenericType* o);

protected:
  /**
   * Resolve an identifier.
   *
   * @tparam ObjectType Object type.
   *
   * @param o The identifier.
   * @param outer The outermost category of scopes to include in the
   * search. The innermost scope is always included, regardless of category.
   */
  template<class ObjectType>
  void resolve(ObjectType* o, const ScopeCategory outer);

  /**
   * Look up a reference that is syntactically ambiguous in an expression
   * context.
   *
   * @param ref The reference.
   *
   * @return A new, unambiguous, reference.
   */
  Expression* lookup(Identifier<Unknown>* ref);

  /**
   * Look up a reference that is syntactically ambiguous in a type
   * context.
   *
   * @param ref The reference.
   *
   * @return A new, unambiguous, reference.
   */
  Type* lookup(UnknownType* ref);

  /**
   * Instantiate a class type with generic arguments.
   */
  virtual void instantiate(ClassType* o);

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
   * List of scopes for resolution of type members.
   */
  std::list<Scope*> memberScopes;

  /**
   * Stack of classes.
   */
  std::list<Class*> classes;

  /**
   * Wrap class types in pointers?
   */
  bool pointers;

  /*
   * Auxiliary visitors.
   */
  Cloner cloner;
};
}

#include "bi/exception/all.hpp"

template<class ObjectType>
void bi::Resolver::resolve(ObjectType* o, const ScopeCategory outer) {
  if (!memberScopes.empty()) {
    /* use the scope for the current member lookup */
    memberScopes.back()->resolve(o);
    memberScopes.pop_back();
  } else {
    /* use current stack of scopes */
    for (auto iter = scopes.rbegin(); !o->target && iter != scopes.rend();
        ++iter) {
      auto scope = *iter;
      if (iter == scopes.rbegin() || scope->category <= outer) {
        scope->resolve(o);
      }
    }
  }
  if (!o->target) {
    throw UnresolvedException(o);
  }
}
