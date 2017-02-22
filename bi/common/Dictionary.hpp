/**
 * @file
 */
#pragma once

#include "bi/exception/PreviousDeclarationException.hpp"
#include "bi/exception/UnresolvedReferenceException.hpp"

#include <unordered_map>
#include <list>

namespace bi {
/**
 * Dictionary for parameters.
 *
 * @ingroup compiler_common
 *
 * @tparam ParameterType Type of parameters.
 * @tparam ReferenceType Type of references.
 */
template<class ParameterType, class ReferenceType>
class Dictionary {
public:
  typedef std::unordered_map<std::string,ParameterType*> map_type;

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
   * @return The parameter to which the reference can be resolved, or
   * `nullptr` if the parameter cannot be resolved.
   */
  virtual ParameterType* resolve(ReferenceType* ref);

  /**
   * Merge another dictionary into this one.
   */
  virtual void merge(Dictionary<ParameterType,ReferenceType>& o);

  /**
   * Declarations within this scope.
   */
  map_type params;
};
}
