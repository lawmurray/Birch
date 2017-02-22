/**
 * @file
 */
#include "bi/common/OverloadedDictionary.hpp"

#include "bi/expression/FuncParameter.hpp"
#include "bi/expression/FuncReference.hpp"
#include "bi/expression/Dispatcher.hpp"
#include "bi/exception/AmbiguousReferenceException.hpp"
#include "bi/exception/PreviousDeclarationException.hpp"

template<class ParameterType, class ReferenceType, class CompareType>
bool bi::OverloadedDictionary<ParameterType,ReferenceType,CompareType>::contains(
    ParameterType* param) {
  auto iter = params.find(param->name->str());
  return iter != params.end() && iter->second.contains(param);
}

template<class ParameterType, class ReferenceType, class CompareType>
ParameterType* bi::OverloadedDictionary<ParameterType,ReferenceType,
    CompareType>::get(ParameterType* param) {
  /* pre-condition */
  assert(contains(param));

  auto iter = params.find(param->name->str());
  assert(iter != params.end());
  return iter->second.get(param);
}

template<class ParameterType, class ReferenceType, class CompareType>
void bi::OverloadedDictionary<ParameterType,ReferenceType,CompareType>::add(
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

template<class ParameterType, class ReferenceType, class CompareType>
void bi::OverloadedDictionary<ParameterType,ReferenceType,CompareType>::merge(
    OverloadedDictionary<ParameterType,ReferenceType,CompareType>& o) {
  for (auto iter1 = o.params.begin(); iter1 != o.params.end(); ++iter1) {
    for (auto iter2 = iter1->second.begin(); iter2 != iter1->second.end();
        ++iter2) {
      if (!contains(*iter2)) {
        add(*iter2);
      }
    }
  }
}

template<class ParameterType, class ReferenceType, class CompareType>
ParameterType* bi::OverloadedDictionary<ParameterType,ReferenceType,
    CompareType>::resolve(ReferenceType* ref) {
  auto iter1 = params.find(ref->name->str());
  if (iter1 == params.end()) {
    return nullptr;
  } else {
    std::list<ParameterType*> matches;
    iter1->second.match(ref, matches);
    if (matches.size() > 1) {
      throw AmbiguousReferenceException(ref, matches);
    } else if (matches.size() == 1) {
      return matches.front();
    } else {
      return nullptr;
    }
  }
}

/*
 * Explicit instantiations.
 */
template class bi::OverloadedDictionary<bi::FuncParameter,bi::FuncReference,
    bi::definitely>;
template class bi::OverloadedDictionary<bi::Dispatcher,bi::FuncReference,
    bi::possibly>;
