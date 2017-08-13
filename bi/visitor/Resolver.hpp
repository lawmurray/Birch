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
  virtual File* modify(File* file);

  using Modifier::modify;

  virtual Expression* modify(List<Expression>* o);
  virtual Expression* modify(Parentheses* o);
  virtual Expression* modify(Brackets* o);
  virtual Expression* modify(Binary* o);
  virtual Expression* modify(Call* o);
  virtual Expression* modify(BinaryCall* o);
  virtual Expression* modify(UnaryCall* o);
  virtual Expression* modify(Slice* o);
  virtual Expression* modify(Query* o);
  virtual Expression* modify(Get* o);
  virtual Expression* modify(LambdaFunction* o);
  virtual Expression* modify(Span* o);
  virtual Expression* modify(Index* o);
  virtual Expression* modify(Range* o);
  virtual Expression* modify(Member* o);
  virtual Expression* modify(Super* o);
  virtual Expression* modify(This* o);
  virtual Expression* modify(Nil* o);
  virtual Expression* modify(Parameter* o);
  virtual Expression* modify(MemberParameter* o);
  virtual Expression* modify(Identifier<Unknown>* o);
  virtual Expression* modify(Identifier<Parameter>* o);
  virtual Expression* modify(Identifier<MemberParameter>* o);
  virtual Expression* modify(Identifier<GlobalVariable>* o);
  virtual Expression* modify(Identifier<LocalVariable>* o);
  virtual Expression* modify(Identifier<MemberVariable>* o);
  virtual Expression* modify(OverloadedIdentifier<Function>* o);
  virtual Expression* modify(OverloadedIdentifier<Fiber>* o);
  virtual Expression* modify(OverloadedIdentifier<MemberFunction>* o);
  virtual Expression* modify(OverloadedIdentifier<MemberFiber>* o);
  virtual Expression* modify(OverloadedIdentifier<BinaryOperator>* o);
  virtual Expression* modify(OverloadedIdentifier<UnaryOperator>* o);

  virtual Statement* modify(Assignment* o);
  virtual Statement* modify(GlobalVariable* o);
  virtual Statement* modify(LocalVariable* o);
  virtual Statement* modify(MemberVariable* o);
  virtual Statement* modify(Function* o);
  virtual Statement* modify(Fiber* o);
  virtual Statement* modify(Program* o);
  virtual Statement* modify(MemberFunction* o);
  virtual Statement* modify(MemberFiber* o);
  virtual Statement* modify(BinaryOperator* o);
  virtual Statement* modify(UnaryOperator* o);
  virtual Statement* modify(AssignmentOperator* o);
  virtual Statement* modify(ConversionOperator* o);
  virtual Statement* modify(Basic* o);
  virtual Statement* modify(Class* o);
  virtual Statement* modify(Alias* o);
  virtual Statement* modify(Import* o);
  virtual Statement* modify(If* o);
  virtual Statement* modify(For* o);
  virtual Statement* modify(While* o);
  virtual Statement* modify(Assert* o);
  virtual Statement* modify(Return* o);
  virtual Statement* modify(Yield* o);

  virtual Type* modify(IdentifierType* o);
  virtual Type* modify(ClassType* o);
  virtual Type* modify(AliasType* o);
  virtual Type* modify(BasicType* o);

protected:
  /**
   * Get the containing class, if any.
   */
  Class* getClass();

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
   * Generic implementation of modify() for variable identifiers.
   */
  template<class ObjectType>
  Identifier<ObjectType>* modifyVariableIdentifier(
      bi::Identifier<ObjectType>* o);

  /**
   * Generic implementation of modify() for function identifiers.
   */
  template<class ObjectType>
  OverloadedIdentifier<ObjectType>* modifyFunctionIdentifier(
      bi::OverloadedIdentifier<ObjectType>* o);

  /**
   * Stack of containing scopes.
   */
  std::list<Scope*> scopes;

  /**
   * Class stack.
   */
  std::stack<Class*> classes;

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

template<class ObjectType>
bi::Identifier<ObjectType>* bi::Resolver::modifyVariableIdentifier(
    bi::Identifier<ObjectType>* o) {
  Modifier::modify(o);
  resolve(o);
  o->type = o->target->type->accept(&cloner)->accept(this);
  return o;
}

template<class ObjectType>
bi::OverloadedIdentifier<ObjectType>* bi::Resolver::modifyFunctionIdentifier(
    bi::OverloadedIdentifier<ObjectType>* o) {
  Modifier::modify(o);
  resolve(o);
  o->type = new OverloadedType(o->target->params, o->target->returns, o->loc);
  return o;
}
