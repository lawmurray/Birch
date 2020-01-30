/**
 * @file
 */
#pragma once

#include "bi/common/Named.hpp"

#include <unordered_set>

namespace bi {
class Parameter;
class GlobalVariable;
class MemberVariable;
class LocalVariable;
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
   * Is there a variable, function, fiber or operator in this scope that
   * matches the identifier?
   */
  bool lookup(const NamedExpression* o) const;

  /**
   * Is there a type in this scope that matches sthe identifier?
   */
  bool lookup(const NamedType* o) const;

  /**
   * Add declaration to scope.
   *
   * @param o Object.
   */
  void add(Parameter* o);
  void add(GlobalVariable* o);
  void add(MemberVariable* o);
  void add(LocalVariable* o);
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
   * Category of this scope.
   */
  const ScopeCategory category;

private:
  /**
   * Names of variables, functions, fibers and operators in this scope.
   */
  std::unordered_set<std::string> names;

  /**
   * Names of types in thi scope.
   */
  std::unordered_set<std::string> typeNames;
};
}
