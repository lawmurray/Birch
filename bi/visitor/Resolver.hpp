/**
 * @file
 */
#pragma once

#include "bi/visitor/Modifier.hpp"
#include "bi/visitor/Cloner.hpp"
#include "bi/visitor/Annotator.hpp"
#include "bi/expression/all.hpp"
#include "bi/statement/all.hpp"
#include "bi/type/all.hpp"

namespace bi {
/**
 * Type of resolver stages.
 *
 * @internal An enum is not used here, as we wish to use e.g. increment
 * operators on the stages.
 */
using ResolverStage = unsigned;

/**
 * First stage.
 */
static const ResolverStage RESOLVER_TYPER = 1;

/**
 * Second stage.
 */
static const ResolverStage RESOLVER_SUPER = 2;

/**
 * Third stage.
 */
static const ResolverStage RESOLVER_HEADER = 3;

/**
 * Fourth stage.
 */
static const ResolverStage RESOLVER_SOURCE = 4;

/**
 * Resolution complete.
 */
static const ResolverStage RESOLVER_FINISHED = 5;

/**
 * Resolve identifiers, infer types, apply code transformations.
 *
 * @ingroup visitor
 */
class Resolver: public Modifier {
public:
  /**
   * Constructor.
   *
   * @param globalStage The global stage to which resolution is to be
   * performed for now.
   */
  Resolver(const ResolverStage globalStage = RESOLVER_TYPER);

  /**
   * Destructor.
   */
  virtual ~Resolver();

  /**
   * Apply to a package.
   */
  void apply(Package* o);

  /**
   * Apply to anything else.
   *
   * @tparam ObjectType Object type.
   *
   * @param o The object.
   * @param globalScope The global scope.
   */
  template<class ObjectType>
  void apply(ObjectType* o, Scope* globalScope);

  virtual Expression* modify(ExpressionList* o);
  virtual Expression* modify(Parentheses* o);
  virtual Expression* modify(Sequence* o);
  virtual Expression* modify(Binary* o);
  virtual Expression* modify(Cast* o);
  virtual Expression* modify(Call<Unknown>* o);
  virtual Expression* modify(Call<Parameter>* o);
  virtual Expression* modify(Call<LocalVariable>* o);
  virtual Expression* modify(Call<MemberVariable>* o);
  virtual Expression* modify(Call<GlobalVariable>* o);
  virtual Expression* modify(Call<Function>* o);
  virtual Expression* modify(Call<MemberFunction>* o);
  virtual Expression* modify(Call<Fiber>* o);
  virtual Expression* modify(Call<MemberFiber>* o);
  virtual Expression* modify(Call<UnaryOperator>* o);
  virtual Expression* modify(Call<BinaryOperator>* o);
  virtual Expression* modify(Assign* o);
  virtual Expression* modify(Slice* o);
  virtual Expression* modify(Query* o);
  virtual Expression* modify(Get* o);
  virtual Expression* modify(LambdaFunction* o);
  virtual Expression* modify(Span* o);
  virtual Expression* modify(Index* o);
  virtual Expression* modify(Range* o);
  virtual Expression* modify(Member* o);
  virtual Expression* modify(Global* o);
  virtual Expression* modify(Super* o);
  virtual Expression* modify(This* o);
  virtual Expression* modify(Nil* o);
  virtual Expression* modify(Parameter* o);
  virtual Expression* modify(Generic* o);
  virtual Expression* modify(Identifier<Unknown>* o);
  virtual Expression* modify(Identifier<Parameter>* o);
  virtual Expression* modify(Identifier<GlobalVariable>* o);
  virtual Expression* modify(Identifier<MemberVariable>* o);
  virtual Expression* modify(Identifier<LocalVariable>* o);
  virtual Expression* modify(Identifier<ForVariable>* o);
  virtual Expression* modify(Identifier<ParallelVariable>* o);
  virtual Expression* modify(OverloadedIdentifier<Unknown>* o);
  virtual Expression* modify(OverloadedIdentifier<Function>* o);
  virtual Expression* modify(OverloadedIdentifier<Fiber>* o);
  virtual Expression* modify(OverloadedIdentifier<MemberFunction>* o);
  virtual Expression* modify(OverloadedIdentifier<MemberFiber>* o);
  virtual Expression* modify(OverloadedIdentifier<BinaryOperator>* o);
  virtual Expression* modify(OverloadedIdentifier<UnaryOperator>* o);

  virtual Statement* modify(GlobalVariable* o);
  virtual Statement* modify(MemberVariable* o);
  virtual Statement* modify(LocalVariable* o);
  virtual Statement* modify(ForVariable* o);
  virtual Statement* modify(ParallelVariable* o);
  virtual Statement* modify(Function* o);
  virtual Statement* modify(Fiber* o);
  virtual Statement* modify(Program* o);
  virtual Statement* modify(MemberFunction* o);
  virtual Statement* modify(MemberFiber* o);
  virtual Statement* modify(BinaryOperator* o);
  virtual Statement* modify(UnaryOperator* o);
  virtual Statement* modify(AssignmentOperator* o);
  virtual Statement* modify(ConversionOperator* o);
  virtual Statement* modify(Class* o);
  virtual Statement* modify(Basic* o);
  virtual Statement* modify(If* o);
  virtual Statement* modify(For* o);
  virtual Statement* modify(Parallel* o);
  virtual Statement* modify(While* o);
  virtual Statement* modify(DoWhile* o);
  virtual Statement* modify(Assert* o);
  virtual Statement* modify(Return* o);
  virtual Statement* modify(Yield* o);
  virtual Statement* modify(Instantiated<Type>* o);
  virtual Statement* modify(Instantiated<Expression>* o);

  virtual Type* modify(UnknownType* o);
  virtual Type* modify(ClassType* o);
  virtual Type* modify(BasicType* o);
  virtual Type* modify(GenericType* o);
  virtual Type* modify(MemberType* o);

private:
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
   * Resolve a call.
   *
   * @tparam ObjectType Object type.
   *
   * @param o The call.
   */
  template<class ObjectType>
  void resolve(Call<ObjectType>* o);

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
   * Look up the type of the referent of a function call.
   *
   * @param o The call.
   *
   * @return A new, unambiguous, call.
   */
  Expression* lookup(Call<Unknown>* o);

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
   * For an expression of Distribution<Value> type, work out Value and return
   * it.
   */
  static Type* getValueType(const Type* o);

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
   * Return type of current function. Stack as functions may contain lambda
   * functions may contain lambda functions...
   */
  std::list<Type*> returnTypes;

  /**
   * Yield type of current fiber.
   */
  std::list<Type*> yieldTypes;

  /**
   * Stage to which the program has been resolved.
   */
  ResolverStage stage;

  /**
   * Stage to which the program should be resolved.
   */
  ResolverStage globalStage;

  /*
   * Auxiliary visitors.
   */
  Cloner cloner;
  Annotator annotator;

  /**
   * Are we currently in a lambda function?
   */
  int inLambda;

  /**
   * Are we currently in a parallel loop?
   */
  int inParallel;

  /**
   * Are we currently in a fiber?
   */
  int inFiber;

  /**
   * Are we currently in a member expression?
   */
  int inMember;
};
}

#include "bi/exception/all.hpp"

template<class ObjectType>
void bi::Resolver::apply(ObjectType* o, Scope* globalScope) {
  scopes.push_back(globalScope);
  for (stage = RESOLVER_TYPER; stage <= globalStage; ++stage) {
    o->accept(this);
  }
  scopes.pop_back();
}

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

template<class ObjectType>
void bi::Resolver::resolve(Call<ObjectType>* o) {
  o->target = o->single->resolve(o);
}

template<class IdentifierType, class ObjectType>
ObjectType* bi::Resolver::instantiate(IdentifierType* o, ObjectType* target) {
  assert(!target->isInstantiation());

  if (target->isGeneric() != !o->typeArgs->isEmpty()) {
    throw GenericException(o, target);
  } else if (target->isGeneric() && o->typeArgs->isBound()) {
    if (o->typeArgs->width() != target->typeParams->width()) {
      throw GenericException(o, target);
    }
    ObjectType* instantiation = target->getInstantiation(o->typeArgs);
    if (!instantiation) {
      instantiation = dynamic_cast<ObjectType*>(target->accept(&cloner));
      assert(instantiation);
      instantiation->bind(o->typeArgs);
      instantiation->stage = globalStage;
      target->addInstantiation(instantiation);

      /* catch up on resolution for this instantiation */
      Resolver resolver(globalStage);
      resolver.apply(instantiation, scopes.front());
    }
    return instantiation;
  } else {
    return target;
  }
}
