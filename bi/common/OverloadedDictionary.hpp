/**
 * @file
 */
#pragma once

#include "bi/common/Dictionary.hpp"
#include "bi/primitive/poset.hpp"
#include "bi/primitive/signature_less_equal.hpp"
#include "bi/exception/UnresolvedReferenceException.hpp"
#include "bi/exception/AmbiguousReferenceException.hpp"

namespace bi {
/**
 * Dictionary for overloadable parameters.
 *
 * @ingroup compiler_common
 *
 * @tparam ParameterType Parameter type.
 */
template<class ParameterType, class ReferenceType>
class OverloadedDictionary: public Dictionary<ParameterType,ReferenceType> {
public:
  typedef poset<ParameterType*,signature_less_equal> poset_type;
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
   * Resolve reference.
   *
   * @param[in,out] ref The reference.
   *
   * If the reference is resolved, updates the definite target of the
   * reference as well as possible alternatives to be checked at runtime.
   * Otherwise sets the definite target to `nullptr` and possible
   * alterantives to the empty list. Only throws an exception if there are
   * multiple definite targets.
   */
  void resolve(ReferenceType* ref);

  /**
   * Get the parents of a vertex in the partial order.
   */
  template<class Container>
  void parents(ParameterType* param, Container& parents) {
    auto iter = overloaded.find(param->name->str());
    if (iter != overloaded.end()) {
      iter->second.parents(param, parents);
    }
  }

  /**
   * Declarations by partial order.
   */
  map_type overloaded;
};
}
