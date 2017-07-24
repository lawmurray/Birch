/**
 * @file
 */
#pragma once

#include "bi/common/Dictionary.hpp"
#include "bi/common/OverloadedDictionary.hpp"
#include "bi/common/Named.hpp"
#include "bi/primitive/definitely.hpp"
#include "bi/primitive/possibly.hpp"

#include <set>

namespace bi {
class Parameter;
class MemberParameter;
class GlobalVariable;
class LocalVariable;
class MemberVariable;
class Function;
class Coroutine;
class MemberFunction;
class MemberCoroutine;
class Program;
class BinaryOperator;
class UnaryOperator;
class Basic;
class Class;
class Alias;
class Unknown;

template<class ObjectType> class Identifier;
class BasicType;
class ClassType;
class AliasType;
class IdentifierType;

/**
 * Categories of objects for identifier loookups.
 */
enum LookupResult {
  PARAMETER,
  MEMBER_PARAMETER,
  GLOBAL_VARIABLE,
  LOCAL_VARIABLE,
  MEMBER_VARIABLE,
  FUNCTION,
  COROUTINE,
  MEMBER_FUNCTION,
  MEMBER_COROUTINE,
  BASIC,
  CLASS,
  ALIAS,
  UNRESOLVED
};

/**
 * Scope.
 *
 * @ingroup compiler_common
 */
class Scope {
public:
  /**
   * Look up the category for an unknown identifier.
   */
  LookupResult lookup(const Identifier<Unknown>* ref) const;
  LookupResult lookup(const IdentifierType* ref) const;

  /**
   * Add parameter.
   *
   * @param param Parameter.
   */
  void add(Parameter* param);
  void add(MemberParameter* param);
  void add(GlobalVariable* param);
  void add(LocalVariable* param);
  void add(MemberVariable* param);
  void add(Function* param);
  void add(Coroutine* param);
  void add(Program* param);
  void add(MemberFunction* param);
  void add(MemberCoroutine* param);
  void add(BinaryOperator* param);
  void add(UnaryOperator* param);
  void add(Class* param);
  void add(Alias* param);
  void add(Basic* param);

  /**
   * Resolve a reference to a parameter.
   *
   * @param ref Reference to resolve.
   *
   * @return Target of the reference.
   */
  void resolve(Identifier<Parameter>* ref);
  void resolve(Identifier<MemberParameter>* ref);
  void resolve(Identifier<GlobalVariable>* ref);
  void resolve(Identifier<LocalVariable>* ref);
  void resolve(Identifier<MemberVariable>* ref);
  void resolve(Identifier<Function>* ref);
  void resolve(Identifier<Coroutine>* ref);
  void resolve(Identifier<Program>* ref);
  void resolve(Identifier<MemberFunction>* ref);
  void resolve(Identifier<MemberCoroutine>* ref);
  void resolve(Identifier<BinaryOperator>* ref);
  void resolve(Identifier<UnaryOperator>* ref);
  void resolve(BasicType* ref);
  void resolve(ClassType* ref);
  void resolve(AliasType* ref);

  /**
   * Inherit another scope into this scope. This is used to import
   * declarations from a base class into a derived class.
   *
   * @param scope Scope to inherit.
   */
  void inherit(Scope* scope);

  /**
   * Import another scope into this scope. This is used to import
   * declarations from one file into another file.
   *
   * @param scope Scope to inherit.
   */
  void import(Scope* scope);

  /**
   * Base scope.
   */
  std::set<Scope*> bases;

  /*
   * Dictionaries.
   */
  Dictionary<Parameter> parameters;
  Dictionary<MemberParameter> memberParameters;
  Dictionary<GlobalVariable> globalVariables;
  Dictionary<LocalVariable> localVariables;
  Dictionary<MemberVariable> memberVariables;
  OverloadedDictionary<Function,definitely> functions;
  OverloadedDictionary<Coroutine,definitely> coroutines;
  Dictionary<Program> programs;
  OverloadedDictionary<MemberFunction,definitely> memberFunctions;
  OverloadedDictionary<MemberCoroutine,definitely> memberCoroutines;
  OverloadedDictionary<BinaryOperator,definitely> binaryOperators;
  OverloadedDictionary<UnaryOperator,definitely> unaryOperators;
  Dictionary<Basic> basics;
  Dictionary<Class> classes;
  Dictionary<Alias> aliases;

private:
  /**
   * Defer lookup to inherited scopes.
   */
  template<class ReferenceType>
  LookupResult lookupInherit(const ReferenceType* ref) const {
    LookupResult result = UNRESOLVED;
    for (auto iter = bases.begin();
        result == UNRESOLVED && iter != bases.end(); ++iter) {
      result = (*iter)->lookup(ref);
    }
    return result;
  }

  /**
   * Defer resolution to inherited scopes.
   */
  template<class ReferenceType>
  void resolveInherit(ReferenceType* ref) {
    for (auto iter = bases.begin(); !ref->target && iter != bases.end();
        ++iter) {
      (*iter)->resolve(ref);
    }
  }
};
}
