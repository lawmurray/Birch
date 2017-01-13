/**
 * @file
 */
#pragma once

#include "bi/common/Dictionary.hpp"
#include "bi/common/OverloadedDictionary.hpp"
#include "bi/common/Named.hpp"

namespace bi {
class VarParameter;
class FuncParameter;
class ModelParameter;
class ProgParameter;

class VarReference;
class FuncReference;
class ModelReference;
class ProgReference;

/**
 * Scope.
 *
 * @ingroup compiler_common
 */
class Scope {
public:
  /**
   * Does the scope contain the declaration?
   *
   * @param param Declaration.
   *
   * For functions, matching is done by signature. For all others, matching
   * is done by name only.
   */
  bool contains(VarParameter* param);
  bool contains(FuncParameter* param);
  bool contains(ModelParameter* param);
  bool contains(ProgParameter* param);

  /**
   * Add parameter.
   *
   * @param param Parameter.
   */
  void add(VarParameter* param);
  void add(FuncParameter* param);
  void add(ModelParameter* param);
  void add(ProgParameter* param);

  /**
   * Resolve a reference to a parameter.
   *
   * @param ref Reference to resolve.
   *
   * @return Target of the reference.
   */
  VarParameter* resolve(VarReference* ref);
  FuncParameter* resolve(FuncReference* ref);
  ModelParameter* resolve(ModelReference* ref);

  /**
   * Import from another scope into this scope.
   *
   * @param scope Scope to import.
   */
  void import(Scope* scope);

  /**
   * Imported scopes.
   *
   * Raw pointers used here to ensure uniqueness.
   */
  std::set<Scope*> imports;

  /**
   * Overloaded declarations, by name.
   */
  Dictionary<VarParameter,VarReference> vars;
  Dictionary<ModelParameter,ModelReference> models;
  OverloadedDictionary<FuncParameter,FuncReference> funcs;
  Dictionary<ProgParameter,ProgReference> progs;

private:
  /**
   * Defer resolution to imported scopes.
   */
  template<class ParameterType, class ReferenceType>
  ParameterType* resolveDefer(ReferenceType* ref);
};
}

template<class ParameterType, class ReferenceType>
ParameterType* bi::Scope::resolveDefer(ReferenceType* ref) {
  ParameterType* target = nullptr;
  auto iter = imports.begin();
  while (!target && iter != imports.end()) {
    target = (*iter)->resolve(ref);
    ++iter;
  }
  return target;
}
