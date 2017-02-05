/**
 * @file
 */
#pragma once

#include "bi/common/Dictionary.hpp"
#include "bi/primitive/poset.hpp"
#include "bi/primitive/definitely.hpp"
#include "bi/primitive/possibly.hpp"

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
  typedef poset<ParameterType*,definitely> definitely_poset_type;
  typedef poset<ParameterType*,possibly> possibly_poset_type;

  typedef std::unordered_map<std::string,definitely_poset_type> definitely_type;
  typedef std::unordered_map<std::string,possibly_poset_type> possibly_type;

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
  virtual void resolve(ReferenceType* ref);

  /**
   * Get the parents of a vertex in the partial order.
   */
  template<class Container>
  void parents(ParameterType* param, Container& parents) {
    auto iter = definites.find(param->name->str());
    if (iter != definites.end()) {
      iter->second.parents(param, parents);
    }
  }

  /**
   * Declarations by definite order.
   */
  definitely_type definites;

  /**
   * Declarations by possible order.
   */
  possibly_type possibles;
};
}
