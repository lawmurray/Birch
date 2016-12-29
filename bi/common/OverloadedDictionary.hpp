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
   * @return Target of the reference.
   *
   * If the reference is resolved, updates the target of the reference and
   * returns true, otherwise returns false. If there are multiple resolutions
   * that cannot be resolved through prioritisation, throws an exception.
   */
  ParameterType* resolve(const ReferenceType* ref);

  /**
   * Declarations within this outer, stored by partial order based on
   * specialisation. Makes for fast lookup when resolving references.
   */
  map_type overloaded;
};
}
