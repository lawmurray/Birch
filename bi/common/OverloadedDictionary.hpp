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
   * Merge another overloaded dictionary into this one.
   */
  void merge(OverloadedDictionary<ParameterType,CompareType>& o);

  /**
   * Resolve reference.
   *
   * @param ref The reference.
   *
   * @return The most-specific parameter to which the reference can be
   * resolved, or `nullptr` if the parameter cannot be resolved.
   */
  template<class ReferenceType>
  ParameterType* resolve(ReferenceType* ref);

  /**
   * Resolve reference.
   *
   * @param ref The reference.
   * @param[out] targets All matching targets.
   */
  template<class ReferenceType, class Container>
  void resolve(ReferenceType* ref, Container& matches);

  /**
   * Declarations by partial order.
   */
  map_type params;
};
}

#include "bi/exception/AmbiguousReferenceException.hpp"

template<class ParameterType, class CompareType>
template<class ReferenceType>
ParameterType* bi::OverloadedDictionary<ParameterType,CompareType>::resolve(
    ReferenceType* ref) {
  std::list<ParameterType*> matches;
  auto iter = params.find(ref->name->str());
  if (iter != params.end()) {
    iter->second.match(ref, matches);
  }
  if (matches.size() > 1) {
    throw AmbiguousReferenceException(ref, matches);
  } else if (matches.size() == 1) {
    return matches.front();
  } else {
    return nullptr;
  }
}

template<class ParameterType, class CompareType>
template<class ReferenceType, class Container>
void bi::OverloadedDictionary<ParameterType,CompareType>::resolve(
    ReferenceType* ref, Container& matches) {
  auto iter = params.find(ref->name->str());
  if (iter != params.end()) {
    iter->second.match_all(ref, matches);
  }
}
