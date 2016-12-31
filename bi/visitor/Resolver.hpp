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

  virtual Expression* modify(ExpressionList* o);
  virtual Expression*  modify(ParenthesesExpression* o);
  virtual Expression*  modify(Range* o);
  virtual Expression*  modify(Traversal* o);
  virtual Expression*  modify(This* o);
  virtual Expression*  modify(BracketsExpression* o);

  virtual Expression*  modify(VarReference* o);
  virtual Expression*  modify(FuncReference* o);
  virtual Expression*  modify(RandomReference* o);
  virtual Type* modify(ModelReference* o);

  virtual Expression*  modify(VarParameter* o);
  virtual Expression*  modify(FuncParameter* o);
  virtual Expression*  modify(RandomParameter* o);
  virtual Prog* modify(ProgParameter* o);
  virtual Type* modify(ModelParameter* o);

  virtual Statement* modify(Import* o);
  virtual Statement* modify(VarDeclaration* o);
  virtual Statement* modify(Conditional* o);
  virtual Statement* modify(Loop* o);

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
