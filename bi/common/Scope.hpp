/**
 * @file
 */
#pragma once

#include "bi/common/Dictionary.hpp"
#include "bi/common/OverloadedDictionary.hpp"
#include "bi/common/Named.hpp"

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
template<class ObjectType> class OverloadedIdentifier;

class BasicType;
class ClassType;
class AliasType;
class IdentifierType;

/**
 * Categories of objects for identifier lookups.
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
  void resolve(OverloadedIdentifier<Function>* ref);
  void resolve(OverloadedIdentifier<Coroutine>* ref);
  void resolve(Identifier<Program>* ref);
  void resolve(OverloadedIdentifier<MemberFunction>* ref);
  void resolve(OverloadedIdentifier<MemberCoroutine>* ref);
  void resolve(OverloadedIdentifier<BinaryOperator>* ref);
  void resolve(OverloadedIdentifier<UnaryOperator>* ref);
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
  OverloadedDictionary<Function> functions;
  OverloadedDictionary<Coroutine> coroutines;
  Dictionary<Program> programs;
  OverloadedDictionary<MemberFunction> memberFunctions;
  OverloadedDictionary<MemberCoroutine> memberCoroutines;
  OverloadedDictionary<BinaryOperator> binaryOperators;
  OverloadedDictionary<UnaryOperator> unaryOperators;
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
