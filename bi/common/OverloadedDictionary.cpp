/**
 * @file
 */
#include "bi/common/OverloadedDictionary.hpp"

#include "bi/expression/FuncParameter.hpp"
#include "bi/expression/FuncReference.hpp"
#include "bi/expression/Dispatcher.hpp"

template<class ParameterType, class CompareType>
bool bi::OverloadedDictionary<ParameterType,CompareType>::contains(
    ParameterType* param) {
  auto iter = params.find(param->name->str());
  return iter != params.end() && iter->second.contains(param);
}

template<class ParameterType, class CompareType>
ParameterType* bi::OverloadedDictionary<ParameterType,CompareType>::get(
    ParameterType* param) {
  /* pre-condition */
  assert(contains(param));

  auto iter = params.find(param->name->str());
  assert(iter != params.end());
  return iter->second.get(param);
}

template<class ParameterType, class CompareType>
void bi::OverloadedDictionary<ParameterType,CompareType>::add(
    ParameterType* param) {
  /* pre-condition */
  assert(!contains(param));

  auto key1 = param->name->str();
  auto iter1 = params.find(key1);
  if (iter1 != params.end()) {
    auto& val1 = iter1->second;
    val1.insert(param);
  } else {
    auto val1 = poset_type();
    val1.insert(param);
    auto pair1 = std::make_pair(key1, val1);
    params.insert(pair1);
  }
}

template<class ParameterType, class CompareType>
void bi::OverloadedDictionary<ParameterType,CompareType>::merge(
    OverloadedDictionary<ParameterType,CompareType>& o) {
  for (auto iter1 = o.params.begin(); iter1 != o.params.end(); ++iter1) {
    for (auto iter2 = iter1->second.begin(); iter2 != iter1->second.end();
        ++iter2) {
      if (!contains(*iter2)) {
        add(*iter2);
      }
    }
  }
}

/*
 * Explicit instantiations.
 */
template class bi::OverloadedDictionary<bi::FuncParameter,bi::definitely>;
template class bi::OverloadedDictionary<bi::Dispatcher,bi::possibly>;
