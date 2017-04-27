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
class OverloadedSet {
public:
  typedef poset<ParameterType*,CompareType> poset_type;

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
  void merge(OverloadedSet<ParameterType,CompareType>& o);

  /**
   * Declarations by partial order.
   */
  poset_type params;
};
}
