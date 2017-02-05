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
  return unordered.find(param->name->str()) != unordered.end();
}

template<class ParameterType, class ReferenceType>
ParameterType* bi::Dictionary<ParameterType,ReferenceType>::get(
    ParameterType* param) {
  /* pre-condition */
  assert(contains(param));

  return unordered.find(param->name->str())->second;
}

template<class ParameterType, class ReferenceType>
void bi::Dictionary<ParameterType,ReferenceType>::add(ParameterType* param) {
  /* pre-condition */
  assert(!contains(param));

  /* store in unordered map */
  auto result = unordered.insert(std::make_pair(param->name->str(), param));
  assert(result.second);

  /* store in ordered list */
  ordered.push_back(param);
}

template<class ParameterType, class ReferenceType>
void bi::Dictionary<ParameterType,ReferenceType>::resolve(
    ReferenceType* ref) {
  auto iter = unordered.find(ref->name->str());
  if (iter != unordered.end() && ref->definitely(*iter->second)) {
    ref->target = iter->second;
  } else {
    ref->target = nullptr;
  }
  ref->alternatives.clear();
}

template<class ParameterType, class ReferenceType>
void bi::Dictionary<ParameterType,ReferenceType>::merge(
    Dictionary<ParameterType,ReferenceType>& o) {
  for (auto iter = o.ordered.begin(); iter != o.ordered.end(); ++iter) {
    if (!contains(*iter)) {
      add(*iter);
    }
  }
}

/*
 * Explicit instantiations.
 */
template class bi::Dictionary<bi::VarParameter,bi::VarReference>;
template class bi::Dictionary<bi::FuncParameter,bi::FuncReference>;
template class bi::Dictionary<bi::ModelParameter,bi::ModelReference>;
template class bi::Dictionary<bi::ProgParameter,bi::ProgReference>;
