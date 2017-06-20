/**
 * @file
 */
#pragma once

#include "bi/primitive/poset.hpp"

#include <unordered_map>

namespace bi {
/**
 * Dictionary for overloadable parameters.
 *
 * @ingroup compiler_common
 *
 * @tparam ParameterType Type of parameters.
 * @tparam CompareType Type of partial order comparison.
 */
template<class ParameterType, class CompareType>
class OverloadedDictionary {
public:
  typedef poset<ParameterType*,CompareType> poset_type;
  typedef std::unordered_map<std::string,poset_type> map_type;

  /**
   * Does the dictionary contain the given parameter?
   */
  bool contains(ParameterType* param) const;

  /**
   * Does the dictionary contain the given name?
   */
  bool contains(const std::string& name) const;

  /**
   * Add parameter.
   *
   * @param param The parameter.
   */
  void add(ParameterType* param);

  /**
   * Import another overloaded dictionary into this one.
   */
  void import(OverloadedDictionary<ParameterType,CompareType>& o);

  /**
   * Resolve reference.
   *
   * @param ref The reference.
   */
  template<class ReferenceType>
  void resolve(ReferenceType* ref);

  /**
   * Declarations by partial order.
   */
  map_type params;
};
}

template<class ParameterType, class CompareType>
template<class ReferenceType>
void bi::OverloadedDictionary<ParameterType,CompareType>::resolve(
    ReferenceType* ref) {
  auto iter = params.find(ref->name->str());
  if (iter != params.end()) {
    iter->second.match(ref, ref->matches);
  }
}
