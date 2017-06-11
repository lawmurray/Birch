/**
 * @file
 */
#pragma once

#include "bi/visitor/Modifier.hpp"
#include "bi/visitor/Cloner.hpp"
#include "bi/visitor/Assigner.hpp"
#include "bi/primitive/shared_ptr.hpp"

#include <stack>

namespace bi {
/**
 * Visitor to resolve references and infer types.
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
  virtual ~Resolver();

  /**
   * Resolve a file.
   */
  virtual void modify(File* file);

  using Modifier::modify;

  virtual Expression* modify(List<Expression>* o);
  virtual Expression*  modify(ParenthesesExpression* o);
  virtual Expression*  modify(Span* o);
  virtual Expression*  modify(Index* o);
  virtual Expression*  modify(Range* o);
  virtual Expression*  modify(Member* o);
  virtual Expression*  modify(Super* o);
  virtual Expression*  modify(This* o);
  virtual Expression*  modify(BracketsExpression* o);

  virtual Expression*  modify(VarReference* o);
  virtual Expression*  modify(FuncReference* o);
  virtual Expression*  modify(BinaryReference* o);
  virtual Expression*  modify(UnaryReference* o);
  virtual Type* modify(TypeReference* o);

  virtual Expression*  modify(VarParameter* o);
  virtual Expression*  modify(FuncParameter* o);
  virtual Expression*  modify(BinaryParameter* o);
  virtual Expression*  modify(UnaryParameter* o);
  virtual Expression*  modify(ConversionParameter* o);
  virtual Expression* modify(ProgParameter* o);
  virtual Type* modify(TypeParameter* o);

  virtual Statement* modify(Import* o);
  virtual Statement* modify(If* o);
  virtual Statement* modify(For* o);
  virtual Statement* modify(While* o);
  virtual Statement* modify(Return* o);

protected:
  /**
   * Make a function associated with a variable of function type.
   */
  FuncParameter* makeFunction(VarParameter* o);

  /**
   * Make a coroutine associated with a variable of coroutine type.
   */
  FuncParameter* makeCoroutine(VarParameter* o);

  /**
   * Take the membership scope, if it exists.
   *
   * @return The membership scope, or nullptr if there is no membership scope
   * at present.
   */
  Scope* takeMemberScope();

  /**
   * Top of the stack of containing scopes.
   */
  Scope* top();

  /**
   * Bottom of the stack of containing scopes.
   */
  Scope* bottom();

  /**
   * Push a scope on the stack of containing scopes.
   *
   * @param scope Scope.
   *
   * If @p scope is @c nullptr, a new scope is created.
   */
  void push(Scope* scope = nullptr);

  /**
   * Pop a scope from the stack of containing scopes.
   */
  Scope* pop();

  /**
   * Resolve a variable reference.
   *
   * @param ref The reference.
   * @param scope The membership scope, if it is to be used for lookup,
   * otherwise the containing scope is used.
   */
  void resolve(VarReference* ref, Scope* scope = nullptr);

  /**
   * Resolve a function reference.
   *
   * @param ref The reference.
   * @param scope The membership scope, if it is to be used for lookup,
   * otherwise the containing scope is used.
   */
  void resolve(FuncReference* ref, Scope* scope = nullptr);

  /**
   * Resolve a reference.
   *
   * @tparam Reference Reference type.
   *
   * @param ref The reference.
   */
  template<class Reference>
  void resolve(Reference* ref, Scope* scope = nullptr);

  /**
   * Defer visit.
   *
   * @param o Braces to which to defer visit.
   */
  void defer(Expression* o);

  /**
   * End deferred visits to the bodies of functions, visiting the bodies of
   * all functions registered since starting.
   */
  void undefer();

  /**
   * Innermost type.
   */
  TypeParameter* type();

  /**
   * Stack of containing scopes.
   */
  std::list<Scope*> scopes;

  /**
   * Type stack.
   */
  std::stack<TypeParameter*> types;

  /**
   * File stack.
   */
  std::stack<File*> files;

  /**
   * Scope for resolution of type members.
   */
  Scope* memberScope;

  /**
   * Deferred functions, binary and unary operators.
   */
  std::list<std::tuple<Expression*,Scope*,TypeParameter*> > defers;

  /**
   * Are we in the input parameters of a function?
   */
  int inInputs;

  /*
   * Auxiliary visitors.
   */
  Cloner cloner;
  Assigner assigner;
};
}

#include "bi/exception/all.hpp"

template<class Reference>
void bi::Resolver::resolve(Reference* ref, Scope* scope) {
  if (scope) {
    /* use provided scope, usually a membership scope */
    scope->resolve(ref);
  } else {
    /* use current stack of scopes */
    ref->target = nullptr;
    for (auto iter = scopes.rbegin(); !ref->target && iter != scopes.rend();
        ++iter) {
      (*iter)->resolve(ref);
    }
  }
  if (!ref->target) {
    throw UnresolvedReferenceException(ref);
  }
}
