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
 * @tparam ReferenceType Type of references.
 */
template<class ParameterType, class ReferenceType, class CompareType>
class OverloadedDictionary {
public:
  typedef poset<ParameterType*,CompareType> poset_type;
  typedef std::unordered_map<std::string,poset_type> map_type;

  /**
   * Does the dictionary contain the given parameter?
   */
  bool contains(ParameterType* param);

  /**
   * If the dictionary contains the given parameter, retrieve its version.
   */
  ParameterType* get(ParameterType* param);

  /**
   * Parents of a parameter.
   */
  template<class Container>
  void parents(ParameterType* param, Container& parents) {
    auto iter = params.find(param->name->str());
    if (iter != params.end()) {
      iter->second.parents(param, parents);
    }
  }

  /**
   * Add parameter.
   *
   * @param param The parameter.
   */
  void add(ParameterType* param);

  /**
   * Merge another overloaded dictionary into this one.
   */
  void merge(
      OverloadedDictionary<ParameterType,ReferenceType,CompareType>& o);

  /**
   * Resolve reference.
   *
   * @param[in,out] ref The reference.
   *
   * @return The parameter to which the reference can be resolved, or
   * `nullptr` if the parameter cannot be resolved.
   */
  ParameterType* resolve(ReferenceType* ref);

  /**
   * Declarations by partial order.
   */
  map_type params;
};
}
