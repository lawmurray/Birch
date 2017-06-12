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
class VarParameter;
class FuncParameter;
class BinaryParameter;
class UnaryParameter;
class AssignmentParameter;
class ConversionParameter;
class TypeParameter;
class ProgParameter;

class VarReference;
class FuncReference;
class BinaryReference;
class UnaryReference;
class AssignmentReference;
class TypeReference;

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
  bool contains(VarParameter* param);
  bool contains(FuncParameter* param);
  bool contains(BinaryParameter* param);
  bool contains(UnaryParameter* param);
  bool contains(AssignmentParameter* param);
  bool contains(ConversionParameter* param);
  bool contains(TypeParameter* param);
  bool contains(ProgParameter* param);

  /**
   * Add parameter.
   *
   * @param param Parameter.
   */
  void add(VarParameter* param);
  void add(FuncParameter* param);
  void add(BinaryParameter* param);
  void add(UnaryParameter* param);
  void add(AssignmentParameter* param);
  void add(ConversionParameter* param);
  void add(TypeParameter* param);
  void add(ProgParameter* param);

  /**
   * Resolve a reference to a parameter.
   *
   * @param ref Reference to resolve.
   *
   * @return Target of the reference.
   */
  void resolve(VarReference* ref);
  void resolve(FuncReference* ref);
  void resolve(BinaryReference* ref);
  void resolve(UnaryReference* ref);
  void resolve(AssignmentReference* ref);
  void resolve(TypeReference* ref);

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
  Dictionary<VarParameter> vars;
  Dictionary<TypeParameter> types;
  OverloadedDictionary<FuncParameter,definitely> funcs;
  OverloadedDictionary<BinaryParameter,definitely> binaries;
  OverloadedDictionary<UnaryParameter,definitely> unaries;
  OverloadedDictionary<AssignmentParameter,definitely> assigns;
  OverloadedSet<ConversionParameter,definitely> convs;
  Dictionary<ProgParameter> progs;

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
