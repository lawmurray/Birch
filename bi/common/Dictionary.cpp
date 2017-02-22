/**
 * @file
 */
#include "bi/common/Dictionary.hpp"

#include "bi/expression/all.hpp"
#include "bi/statement/all.hpp"
#include "bi/type/all.hpp"
#include "bi/program/all.hpp"
#include "bi/exception/all.hpp"

template<class ParameterType, class ReferenceType>
bool bi::Dictionary<ParameterType,ReferenceType>::contains(
    ParameterType* param) {
  return params.find(param->name->str()) != params.end();
}

template<class ParameterType, class ReferenceType>
ParameterType* bi::Dictionary<ParameterType,ReferenceType>::get(
    ParameterType* param) {
  /* pre-condition */
  assert(contains(param));

  return params.find(param->name->str())->second;
}

template<class ParameterType, class ReferenceType>
void bi::Dictionary<ParameterType,ReferenceType>::add(ParameterType* param) {
  /* pre-condition */
  assert(!contains(param));

  auto result = params.insert(std::make_pair(param->name->str(), param));
  assert(result.second);
}

template<class ParameterType, class ReferenceType>
ParameterType* bi::Dictionary<ParameterType,ReferenceType>::resolve(
    ReferenceType* ref) {
  auto iter = params.find(ref->name->str());
  if (iter != params.end() && ref->definitely(*iter->second)) {
    return iter->second;
  } else {
    return nullptr;
  }
}

template<class ParameterType, class ReferenceType>
void bi::Dictionary<ParameterType,ReferenceType>::merge(
    Dictionary<ParameterType,ReferenceType>& o) {
  for (auto iter = o.params.begin(); iter != o.params.end(); ++iter) {
    if (!contains(iter->second)) {
      add(iter->second);
    }
  }
}

/*
 * Explicit instantiations.
 */
template class bi::Dictionary<bi::VarParameter,bi::VarReference>;
template class bi::Dictionary<bi::ModelParameter,bi::ModelReference>;
template class bi::Dictionary<bi::ProgParameter,bi::ProgReference>;
