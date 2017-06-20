/**
 * @file
 */
#pragma once

#include "bi/visitor/Modifier.hpp"
#include "bi/visitor/Cloner.hpp"
#include "bi/visitor/Assigner.hpp"
#include "bi/primitive/shared_ptr.hpp"

#include <stack>
#include <list>

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
  virtual Expression* modify(ParenthesesExpression* o);
  virtual Expression* modify(BracketsExpression* o);
  virtual Expression* modify(LambdaFunction* o);
  virtual Expression* modify(Span* o);
  virtual Expression* modify(Index* o);
  virtual Expression* modify(Range* o);
  virtual Expression* modify(Member* o);
  virtual Expression* modify(Super* o);
  virtual Expression* modify(This* o);
  virtual Expression* modify(Parameter* o);
  virtual Expression* modify(Identifier<Unknown>* o);
  virtual Expression* modify(Identifier<Parameter>* o);
  virtual Expression* modify(Identifier<GlobalVariable>* o);
  virtual Expression* modify(Identifier<LocalVariable>* o);
  virtual Expression* modify(Identifier<MemberVariable>* o);
  virtual Expression* modify(Identifier<Function>* o);
  virtual Expression* modify(Identifier<Coroutine>* o);
  virtual Expression* modify(Identifier<MemberFunction>* o);
  virtual Expression* modify(Identifier<BinaryOperator>* o);
  virtual Expression* modify(Identifier<UnaryOperator>* o);

  virtual Statement* modify(Assignment* o);
  virtual Statement* modify(GlobalVariable* o);
  virtual Statement* modify(LocalVariable* o);
  virtual Statement* modify(MemberVariable* o);
  virtual Statement* modify(Function* o);
  virtual Statement* modify(Coroutine* o);
  virtual Statement* modify(Program* o);
  virtual Statement* modify(MemberFunction* o);
  virtual Statement* modify(BinaryOperator* o);
  virtual Statement* modify(UnaryOperator* o);
  virtual Statement* modify(AssignmentOperator* o);
  virtual Statement* modify(ConversionOperator* o);
  virtual Statement* modify(Class* o);
  virtual Statement* modify(Alias* o);
  virtual Statement* modify(Basic* o);
  virtual Statement* modify(Import* o);
  virtual Statement* modify(If* o);
  virtual Statement* modify(For* o);
  virtual Statement* modify(While* o);
  virtual Statement* modify(Return* o);

  virtual Type* modify(IdentifierType* o);
  virtual Type* modify(ClassType* o);
  virtual Type* modify(AliasType* o);
  virtual Type* modify(BasicType* o);

protected:
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
   * Resolve a reference.
   *
   * @tparam Reference Reference type.
   *
   * @param ref The reference.
   * @param scope The membership scope, if it is to be used for lookup,
   * otherwise the containing scope is used.
   */
  template<class Reference>
  void resolve(Reference* ref, Scope* scope = nullptr);

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
  Type* lookup(IdentifierType* ref, Scope* scope = nullptr);

  /**
   * Defer visit.
   *
   * @param o Braces to which to defer visit.
   */
  void defer(Statement* o);

  /**
   * End deferred visits to the bodies of functions, visiting the bodies of
   * all functions registered since starting.
   */
  void undefer();

  /**
   * Innermost type.
   */
  Class* type();

  /**
   * Stack of containing scopes.
   */
  std::list<Scope*> scopes;

  /**
   * Type stack.
   */
  std::stack<Class*> types;

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
  std::list<std::tuple<Statement*,Scope*,Class*> > defers;

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
    for (auto iter = scopes.rbegin();
        ref->matches.size() == 0 && iter != scopes.rend(); ++iter) {
      (*iter)->resolve(ref);
    }
  }
  if (ref->matches.size() == 0) {
    throw UnresolvedReferenceException(ref);
  } else if (ref->matches.size() > 1) {
    throw AmbiguousReferenceException(ref);
  } else {
    ref->target = ref->matches.front();
  }
}
