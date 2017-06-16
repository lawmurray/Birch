/**
 * @file
 */
#pragma once

#include "bi/common/Dictionary.hpp"
#include "bi/common/OverloadedDictionary.hpp"
#include "bi/common/OverloadedSet.hpp"
#include "bi/common/Named.hpp"
#include "bi/primitive/definitely.hpp"
#include "bi/primitive/possibly.hpp"

#include <set>

namespace bi {
class Parameter;
class GlobalVariable;
class LocalVariable;
class MemberVariable;
class Function;
class Coroutine;
class MemberFunction;
class Program;
class BinaryOperator;
class UnaryOperator;
class AssignmentOperator;
class ConversionOperator;
class Class;
class AliasType;
class BasicType;

template<class ObjectType> class Identifier;
template<class ObjectType> class IdentifierType;
class Assignment;

/**
 * Scope.
 *
 * @ingroup compiler_common
 */
class Scope {
public:
  /**
   * Does the scope contain the parameter?
   *
   * @param param Parameter.
   *
   * For functions, matching is done by signature. For all others, matching
   * is done by name only.
   */
  bool contains(Parameter* param);
  bool contains(GlobalVariable* param);
  bool contains(LocalVariable* param);
  bool contains(MemberVariable* param);
  bool contains(Function* param);
  bool contains(Coroutine* param);
  bool contains(Program* param);
  bool contains(MemberFunction* param);
  bool contains(BinaryOperator* param);
  bool contains(UnaryOperator* param);
  bool contains(AssignmentOperator* param);
  bool contains(ConversionOperator* param);
  bool contains(Class* param);
  bool contains(AliasType* param);
  bool contains(BasicType* param);

  /**
   * Add parameter.
   *
   * @param param Parameter.
   */
  void add(Parameter* param);
  void add(GlobalVariable* param);
  void add(LocalVariable* param);
  void add(MemberVariable* param);
  void add(Function* param);
  void add(Coroutine* param);
  void add(Program* param);
  void add(MemberFunction* param);
  void add(BinaryOperator* param);
  void add(UnaryOperator* param);
  void add(AssignmentOperator* param);
  void add(ConversionOperator* param);
  void add(Class* param);
  void add(AliasType* param);
  void add(BasicType* param);

  /**
   * Resolve a reference to a parameter.
   *
   * @param ref Reference to resolve.
   *
   * @return Target of the reference.
   */
  void resolve(Identifier<Parameter>* ref);
  void resolve(Identifier<GlobalVariable>* ref);
  void resolve(Identifier<LocalVariable>* ref);
  void resolve(Identifier<MemberVariable>* ref);
  void resolve(Identifier<Function>* ref);
  void resolve(Identifier<Coroutine>* ref);
  void resolve(Identifier<Program>* ref);
  void resolve(Identifier<MemberFunction>* ref);
  void resolve(Identifier<BinaryOperator>* ref);
  void resolve(Identifier<UnaryOperator>* ref);
  void resolve(IdentifierType<Class>* ref);
  void resolve(IdentifierType<AliasType>* ref);
  void resolve(IdentifierType<BasicType>* ref);
  void resolve(Assignment* ref);

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
  Dictionary<GlobalVariable> globalVariables;
  Dictionary<LocalVariable> localVariables;
  Dictionary<MemberVariable> memberVariables;
  OverloadedDictionary<Function,definitely> functions;
  OverloadedDictionary<Coroutine,definitely> coroutines;
  Dictionary<Program> programs;
  OverloadedDictionary<MemberFunction,definitely> memberFunctions;
  OverloadedDictionary<BinaryOperator,definitely> binaryOperators;
  OverloadedDictionary<UnaryOperator,definitely> unaryOperators;
  OverloadedDictionary<AssignmentOperator,definitely> assignmentOperators;
  OverloadedSet<ConversionOperator,definitely> conversionOperators;
  Dictionary<Class> classes;
  Dictionary<AliasType> aliasTypes;
  Dictionary<BasicType> basicTypes;

private:
  /**
   * Defer resolution to imported scopes.
   */
  template<class ReferenceType>
  void resolveDefer(ReferenceType* ref) {
    for (auto iter = bases.begin(); !ref->target && iter != bases.end();
        ++iter) {
      (*iter)->resolve(ref);
    }
  }
};
}
