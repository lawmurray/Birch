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

  virtual Expression* modify(ExpressionList* o);
  virtual Expression*  modify(ParenthesesExpression* o);
  virtual Expression*  modify(Index* o);
  virtual Expression*  modify(Range* o);
  virtual Expression*  modify(Member* o);
  virtual Expression*  modify(This* o);
  virtual Expression*  modify(BracketsExpression* o);

  virtual Expression*  modify(VarReference* o);
  virtual Expression*  modify(FuncReference* o);
  virtual Type* modify(TypeReference* o);

  virtual Expression*  modify(VarParameter* o);
  virtual Expression*  modify(FuncParameter* o);
  virtual Prog* modify(ProgParameter* o);
  virtual Type* modify(TypeParameter* o);

  virtual Statement* modify(Import* o);
  virtual Statement* modify(Conditional* o);
  virtual Statement* modify(Loop* o);
  virtual Statement* modify(Return* o);

protected:
  /**
   * Make a function associated with a variable of lambda type.
   */
  FuncParameter* makeLambda(VarParameter* o);

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
   * Resolve a type reference.
   *
   * @param ref The reference.
   */
  void resolve(TypeReference* ref);

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
