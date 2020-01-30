/**
 * @file
 */
#pragma once

#include "bi/visitor/Modifier.hpp"
#include "bi/visitor/Cloner.hpp"
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
 * Third stage.
 */
static const ResolverStage RESOLVER_HEADER = 1;

/**
 * Fourth stage.
 */
static const ResolverStage RESOLVER_SOURCE = 2;

/**
 * Resolution complete.
 */
static const ResolverStage RESOLVER_FINISHED = 3;

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
  Resolver(const ResolverStage globalStage = RESOLVER_HEADER);

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

  virtual Expression* modify(Assign* o);
  virtual Expression* modify(LambdaFunction* o);
  virtual Expression* modify(NamedExpression* o);

  virtual Statement* modify(GlobalVariable* o);
  virtual Statement* modify(MemberVariable* o);
  virtual Statement* modify(LocalVariable* o);
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

  virtual Type* modify(NamedType* o);

private:
  /**
   * Categorize an identifier.
   *
   * @param o The identifier.
   */
  void lookup(NamedExpression* o);

  /**
   * Categorize an identifier.
   *
   * @param o The identifier.
   */
  void lookup(NamedType* o);

  /**
   * List of scopes, innermost at the back.
   */
  std::list<Scope*> scopes;

  /**
   * Stage to which the program has been resolved.
   */
  ResolverStage stage;

  /**
   * Stage to which the program should be resolved.
   */
  ResolverStage globalStage;

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
  for (stage = RESOLVER_HEADER; stage <= globalStage; ++stage) {
    o->accept(this);
  }
  scopes.pop_back();
}
