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
  bool contains(ParameterType* param) const;

  /**
   * Does the dictionary contain the given parameter?
   */
  bool contains(const std::string& name) const;

  /**
   * Get a parameter by name.
   */
  ParameterType* get(const std::string& name);

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
  void resolve(ReferenceType* ref);

  /**
   * Import another dictionary into this one.
   */
  void import(Dictionary<ParameterType>& o);

  /**
   * Declarations within this scope.
   */
  map_type params;
};
}

template<class ParameterType>
template<class ReferenceType>
void bi::Dictionary<ParameterType>::resolve(ReferenceType* ref) {
  auto iter = params.find(ref->name->str());
  if (iter != params.end()) {
    ref->matches.push_back(iter->second);
  }
}
