/**
 * @file
 */
#pragma once

#include "bi/visitor/Modifier.hpp"
#include "bi/visitor/Cloner.hpp"
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
   *
   * @param scope Root scope.
   */
  Resolver(shared_ptr<Scope> scope = new Scope());

  /**
   * Destructor.
   */
  virtual ~Resolver();

  /**
   * Resolve a file.
   */
  virtual void modify(File* file);

  virtual void modify(ExpressionList* o);
  virtual void modify(ParenthesesExpression* o);
  virtual void modify(BracesExpression* o);
  virtual void modify(RandomVariable* o);
  virtual void modify(Range* o);
  virtual void modify(Traversal* o);
  virtual void modify(This* o);
  virtual void modify(BracketsExpression* o);

  virtual void modify(VarReference* o);
  virtual void modify(FuncReference* o);
  virtual void modify(ModelReference* o);

  virtual void modify(VarParameter* o);
  virtual void modify(FuncParameter* o);
  virtual void modify(ProgParameter* o);
  virtual void modify(ModelParameter* o);

  virtual void modify(Import* o);
  virtual void modify(ExpressionStatement* o);
  virtual void modify(VarDeclaration* o);
  virtual void modify(Conditional* o);
  virtual void modify(Loop* o);

protected:
  /**
   * Innermost scope.
   */
  shared_ptr<Scope> inner();

  /**
   * Push a scope on the stack.
   *
   * @param scope Scope.
   *
   * If @p scope is @c nullptr, a new scope is created.
   */
  void push(shared_ptr<Scope> scope = nullptr);

  /**
   * Pop a scope from the stack.
   */
  shared_ptr<Scope> pop();

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
   * Innermost model.
   */
  ModelParameter* model();

  /**
   * Scope stack.
   */
  std::stack<shared_ptr<Scope>> scopes;

  /**
   * Model stack.
   */
  std::stack<ModelParameter*> models;

  /**
   * File stack.
   */
  std::stack<File*> files;

  /**
   * Scope for traversing model members.
   */
  shared_ptr<Scope> traverseScope;

  /**
   * Deferred functions, binary and unary operators.
   */
  std::list<std::tuple<Expression*,shared_ptr<Scope>,ModelParameter*> > defers;

  /**
   * Are we in the input parameters of a function?
   */
  bool inInputs;

  /**
   * Auxiliary visitors.
   */
  Cloner cloner;
};
}
