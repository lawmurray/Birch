/**
 * @file
 */
#pragma once

#include <unordered_map>
#include <string>

namespace bi {
/**
 * Dictionary for parameters.
 *
 * @ingroup compiler_common
 *
 * @tparam ParameterType Type of parameters.
 */
template<class ParameterType>
class Dictionary {
public:
  typedef std::unordered_map<std::string,ParameterType*> map_type;

  /**
   * Does the dictionary contain the given parameter?
   */
  bool contains(ParameterType* param);

  /**
   * If the dictionary contains the given parameter, retrieve its version.
   */
  ParameterType* get(ParameterType* param);

  /**
   * Add parameter.
   *
   * @param param The parameter.
   */
  void add(ParameterType* param);

  /**
   * Resolve reference.
   *
   * @param[in,out] ref The reference.
   *
   * @return The parameter to which the reference can be resolved, or
   * `nullptr` if the parameter cannot be resolved.
   */
  template<class ReferenceType>
  ParameterType* resolve(ReferenceType* ref);

  /**
   * Merge another dictionary into this one.
   */
  void merge(Dictionary<ParameterType>& o);

  /**
   * Declarations within this scope.
   */
  map_type params;
};
}

template<class ParameterType>
template<class ReferenceType>
ParameterType* bi::Dictionary<ParameterType>::resolve(ReferenceType* ref) {
  auto iter = params.find(ref->name->str());
  if (iter != params.end() && ref->definitely(*iter->second)) {
    return iter->second;
  } else {
    return nullptr;
  }
}
