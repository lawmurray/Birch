/**
 * @file
 */
#pragma once

#include "bi/visitor/Modifier.hpp"
#include "bi/visitor/Cloner.hpp"

#include <stack>
#include <list>

namespace bi {
/**
 * Abstract class for resolvers, with common functionality.
 *
 * @ingroup birch_visitor
 */
class Resolver: public Modifier {
public:
  /**
   * Constructor.
   *
   * @param rootScope The root scope.
   */
  Resolver(Scope* rootScope);

  /**
   * Destructor.
   */
  virtual ~Resolver() = 0;

  using Modifier::modify;

  virtual Expression* modify(ExpressionList* o);
  virtual Expression* modify(Parentheses* o);
  virtual Expression* modify(Sequence* o);
  virtual Expression* modify(Binary* o);

  virtual Type* modify(TypeIdentifier* o);
  virtual Type* modify(ClassType* o);
  virtual Type* modify(AliasType* o);
  virtual Type* modify(BasicType* o);
  virtual Type* modify(GenericType* o);

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
  Type* lookup(TypeIdentifier* ref);

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

  /*
   * Auxiliary visitors.
   */
  Cloner cloner;
};
}

#include "bi/exception/all.hpp"

template<class ObjectType>
void bi::Resolver::resolve(ObjectType* o) {
  if (!memberScopes.empty()) {
    /* use the scope for the current member lookup */
    memberScopes.back()->resolve(o);
    memberScopes.pop_back();
  } else {
    /* use current stack of scopes */
    for (auto iter = scopes.rbegin(); !o->target && iter != scopes.rend();
        ++iter) {
      (*iter)->resolve(o);
    }
  }
  if (!o->target) {
    assert(false);
    throw UnresolvedException(o);
  }
}
