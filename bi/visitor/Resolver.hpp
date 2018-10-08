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
  virtual Type* modify(MemberType* o);

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
   * Instantiate a generic class or function.
   *
   * @tparam ObjectType Object type.
   *
   * @param o The identifier.
   */
  template<class IdentifierType, class ObjectType>
  ObjectType* instantiate(IdentifierType* o, ObjectType* target);

  /**
   * Look up an identifier that is syntactically ambiguous.
   *
   * @param o The identifier.
   *
   * @return A new, unambiguous, identifier.
   */
  Expression* lookup(Identifier<Unknown>* o);

  /**
   * Look up an identifier that is syntactically ambiguous.
   *
   * @param o The identifier.
   *
   * @return A new, unambiguous, identifier.
   */
  Expression* lookup(OverloadedIdentifier<Unknown>* o);

  /**
   * Look up a type identifier that is syntactically ambiguous.
   *
   * @param o The identifier.
   *
   * @return A new, unambiguous, type identifier.
   */
  Type* lookup(UnknownType* o);

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

template<class IdentifierType, class ObjectType>
ObjectType* bi::Resolver::instantiate(IdentifierType* o, ObjectType* target) {
  if (target->isGeneric() && o->typeArgs->isBound()) {
    if (o->typeArgs->width() != target->typeParams->width()) {
      throw GenericException(o, target);
    }
    auto instantiation = target->getInstantiation(o->typeArgs);
    if (!instantiation) {
      instantiation = dynamic_cast<decltype(instantiation)>(target->accept(&cloner));
      assert(instantiation);
      instantiation->bind(o->typeArgs);
      target->addInstantiation(instantiation);
      instantiation->accept(this);
    }
    return instantiation;
  } else {
    return target;
  }
}
