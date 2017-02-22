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
  virtual bool contains(ParameterType* param);

  /**
   * If the dictionary contains the given parameter, retrieve its version.
   */
  virtual ParameterType* get(ParameterType* param);

  /**
   * Add parameter.
   *
   * @param param The parameter.
   */
  virtual void add(ParameterType* param);

  /**
   * Merge another overloaded dictionary into this one.
   */
  virtual void merge(
      OverloadedDictionary<ParameterType,ReferenceType,CompareType>& o);

  /**
   * Resolve reference.
   *
   * @param[in,out] ref The reference.
   *
   * @return The parameter to which the reference can be resolved, or
   * `nullptr` if the parameter cannot be resolved.
   */
  virtual ParameterType* resolve(ReferenceType* ref);

  /**
   * Declarations by partial order.
   */
  map_type params;
};
}
