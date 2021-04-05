/**
 * @file
 */
#pragma once

#include "src/common/Named.hpp"

namespace birch {
class Parameter;
class GlobalVariable;
class MemberVariable;
class LocalVariable;
class Function;
class MemberFunction;
class BinaryOperator;
class UnaryOperator;
class Program;
class Basic;
class Class;
class Generic;

class NamedExpression;
class NamedType;

/**
 * Scope categories.
 */
enum ScopeCategory {
  LOCAL_SCOPE,
  MEMBER_SCOPE,
  GLOBAL_SCOPE
};

/**
 * Categories for names in the context of an expression.
 */
enum ExpressionCategory {
  UNKNOWN,
  MEMBER_UNKNOWN,
  PARAMETER,
  LOCAL_VARIABLE,
  MEMBER_VARIABLE,
  GLOBAL_VARIABLE,
  MEMBER_FUNCTION,
  GLOBAL_FUNCTION,
  UNARY_OPERATOR,
  BINARY_OPERATOR
};

/**
 * Categories for nameas in the context of a type.
 */
enum TypeCategory {
  UNKNOWN_TYPE,
  BASIC_TYPE,
  STRUCT_TYPE,
  CLASS_TYPE,
  GENERIC_TYPE
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
   *
   * @param Category of this scope.
   */
  Scope(const ScopeCategory category);

  /**
   * Look up a name in the context of an expression.
   *
   * If the name is found in the current scope, updates `o->category` and
   * `o->number` accordingly.
   */
  void lookup(NamedExpression* o) const;

  /**
   * Look up name in the context of a type.
   *
   * If the name is found in the current scope, updates `o->category` and
   * `o->number` accordingly.
   */
  void lookup(NamedType* o) const;

  /**
   * Resolve the inheritance of a class. Looks up the base class of the `o`
   * in this scope. If found, updates `o->scope->base` to the scope of that
   * class.
   */
  void inherit(Class* o) const;

  /**
   * Add declaration to scope.
   *
   * @param o Object.
   */
  void add(Parameter* o);
  void add(LocalVariable* o);
  void add(MemberVariable* o);
  void add(GlobalVariable* o);
  void add(MemberFunction* o);
  void add(Function* o);
  void add(BinaryOperator* o);
  void add(UnaryOperator* o);
  void add(Program* o);
  void add(Basic* o);
  void add(Class* o);
  void add(Generic* o);

  /**
   * Check if the scope overrides a declaration of the given name in a base
   * class.
   */
  bool overrides(const std::string& name) const;

  /**
   * Base scope.
   */
  Scope* base;

  /**
   * Category of this scope.
   */
  const ScopeCategory category;

  /*
   * Variables, functions, operators, etc in this scope.
   */
  std::unordered_multimap<std::string,Parameter*> parameters;
  std::unordered_multimap<std::string,LocalVariable*> localVariables;
  std::unordered_multimap<std::string,MemberVariable*> memberVariables;
  std::unordered_multimap<std::string,GlobalVariable*> globalVariables;
  std::unordered_multimap<std::string,MemberFunction*> memberFunctions;
  std::unordered_multimap<std::string,Function*> functions;
  std::unordered_multimap<std::string,BinaryOperator*> binaryOperators;
  std::unordered_multimap<std::string,UnaryOperator*> unaryOperators;
  std::unordered_multimap<std::string,Program*> programs;

  /*
   * Types in this scope.
   */
  std::unordered_multimap<std::string,Basic*> basicTypes;
  std::unordered_multimap<std::string,Class*> classTypes;
  std::unordered_multimap<std::string,Generic*> genericTypes;
};
}
