/**
 * @file
 */
#include "bi/common/OverloadedSet.hpp"

#include "bi/statement/ConversionOperator.hpp"

template<class ParameterType, class CompareType>
bool bi::OverloadedSet<ParameterType,CompareType>::contains(
    ParameterType* param) {
  return params.contains(param);
}

template<class ParameterType, class CompareType>
ParameterType* bi::OverloadedSet<ParameterType,CompareType>::get(
    ParameterType* param) {
  /* pre-condition */
  assert(contains(param));

  return params.get(param);
}

template<class ParameterType, class CompareType>
void bi::OverloadedSet<ParameterType,CompareType>::add(ParameterType* param) {
  /* pre-condition */
  assert(!contains(param));

  params.insert(param);
}

template<class ParameterType, class CompareType>
void bi::OverloadedSet<ParameterType,CompareType>::merge(
    OverloadedSet<ParameterType,CompareType>& o) {
  for (auto iter = o.params.begin(); iter != o.params.end(); ++iter) {
    if (!contains(*iter)) {
      add(*iter);
    }
  }
}

/*
 * Explicit instantiations.
 */
template class bi::OverloadedSet<bi::ConversionOperator,bi::definitely>;
