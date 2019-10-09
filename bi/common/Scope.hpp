/**
 * @file
 */
#pragma once

#include "bi/common/Dictionary.hpp"
#include "bi/common/OverloadedDictionary.hpp"
#include "bi/common/Named.hpp"
#include "bi/common/Lookup.hpp"

namespace bi {
class Parameter;
class LocalVariable;
class MemberVariable;
class GlobalVariable;
class Function;
class MemberFunction;
class Fiber;
class MemberFiber;
class BinaryOperator;
class UnaryOperator;
class Program;
class Basic;
class Class;
class Generic;
class Unknown;

template<class ObjectType> class Identifier;
template<class ObjectType> class OverloadedIdentifier;

class BasicType;
class ClassType;
class GenericType;
class UnknownType;

/**
 * Scope categories.
 */
enum ScopeCategory {
  LOCAL_SCOPE,
  CLASS_SCOPE,
  GLOBAL_SCOPE
};

/**
 * Scope.
 *
 * @ingroup common
 */
class Scope {
public:
  /**
   * Constructor.
   */
  Scope(const ScopeCategory category);

  /**
   * Look up the category for an unknown identifier.
   */
  Lookup lookup(const Identifier<Unknown>* o) const;

  /**
   * Look up the category for an unknown identifier.
   */
  Lookup lookup(const OverloadedIdentifier<Unknown>* o) const;

  /**
   * Look up the category for an unknown type.
   */
  Lookup lookup(const UnknownType* o) const;

  /**
   * Add declaration to scope.
   *
   * @param o Object.
   */
  void add(Parameter* o);
  void add(GlobalVariable* o);
  void add(LocalVariable* o);
  void add(MemberVariable* o);
  void add(Function* o);
  void add(Fiber* o);
  void add(Program* o);
  void add(MemberFunction* o);
  void add(MemberFiber* o);
  void add(BinaryOperator* o);
  void add(UnaryOperator* o);
  void add(Basic* o);
  void add(Class* o);
  void add(Generic* o);

  /**
   * Get the declaration to which an identifier corresponds.
   *
   * @param o Identifier.
   *
   * @return Declaration.
   */
  void resolve(Identifier<Parameter>* o);
  void resolve(Identifier<GlobalVariable>* o);
  void resolve(Identifier<LocalVariable>* o);
  void resolve(Identifier<MemberVariable>* o);
  void resolve(OverloadedIdentifier<Function>* o);
  void resolve(OverloadedIdentifier<Fiber>* o);
  void resolve(Identifier<Program>* o);
  void resolve(OverloadedIdentifier<MemberFunction>* o);
  void resolve(OverloadedIdentifier<MemberFiber>* o);
  void resolve(OverloadedIdentifier<BinaryOperator>* o);
  void resolve(OverloadedIdentifier<UnaryOperator>* o);
  void resolve(BasicType* o);
  void resolve(ClassType* o);
  void resolve(GenericType* o);

  /**
   * Inherit another scope into this scope. This is used to inherit
   * declarations from a base class into a derived class.
   *
   * @param scope Scope to inherit.
   */
  void inherit(Scope* scope);

  /**
   * Check if an object overrides another in a base class (by having the same
   * name, not necessarily the same signature).
   */
  bool override(const MemberFunction* o) const;
  bool override(const MemberFiber* o) const;

  /**
   * Base scope.
   */
  Scope* base;

  /**
   * Scope category.
   */
  ScopeCategory category;

  /*
   * Dictionaries.
   */
  Dictionary<Parameter> parameters;
  Dictionary<GlobalVariable> globalVariables;
  Dictionary<LocalVariable> localVariables;
  Dictionary<MemberVariable> memberVariables;
  OverloadedDictionary<Function> functions;
  OverloadedDictionary<Fiber> fibers;
  Dictionary<Program> programs;
  OverloadedDictionary<MemberFunction> memberFunctions;
  OverloadedDictionary<MemberFiber> memberFibers;
  OverloadedDictionary<BinaryOperator> binaryOperators;
  OverloadedDictionary<UnaryOperator> unaryOperators;
  Dictionary<Basic> basics;
  Dictionary<Class> classes;
  Dictionary<Generic> generics;

private:
  /**
   * Defer lookup to inherited scopes.
   */
  Lookup lookupInherit(const Identifier<Unknown>* o) const;

  /**
   * Defer lookup to inherited scopes.
   */
  Lookup lookupInherit(const OverloadedIdentifier<Unknown>* o) const;

  /**
   * Defer lookup to inherited scopes.
   */
  Lookup lookupInherit(const UnknownType* o) const;

  /**
   * Check for previous declarations of the same name, at global scope.
   */
  template<class ParameterType>
  void checkPreviousGlobal(ParameterType* param);

  /**
   * Check for previous declarations of the same name, at local scope.
   */
  template<class ParameterType>
  void checkPreviousLocal(ParameterType* param);

  /**
   * Check for previous declarations of the same name, for a member.
   */
  template<class ParameterType>
  void checkPreviousMember(ParameterType* param);

  /**
   * Check for previous declarations of the same name, for a type.
   */
  template<class ParameterType>
  void checkPreviousType(ParameterType* param);
};
}
