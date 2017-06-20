/**
 * @file
 */
#include "bi/common/OverloadedDictionary.hpp"

#include "bi/statement/Function.hpp"
#include "bi/statement/Coroutine.hpp"
#include "bi/statement/MemberFunction.hpp"
#include "bi/statement/BinaryOperator.hpp"
#include "bi/statement/UnaryOperator.hpp"
#include "bi/statement/AssignmentOperator.hpp"

template<class ParameterType, class CompareType>
bool bi::OverloadedDictionary<ParameterType,CompareType>::contains(
    ParameterType* param) const {
  auto iter = params.find(param->name->str());
  return iter != params.end() && iter->second.contains(param);
}

template<class ParameterType, class CompareType>
bool bi::OverloadedDictionary<ParameterType,CompareType>::contains(
    const std::string& name) const {
  return params.find(name) != params.end();
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
void bi::OverloadedDictionary<ParameterType,CompareType>::import(
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
template class bi::OverloadedDictionary<bi::Function,bi::definitely>;
template class bi::OverloadedDictionary<bi::Coroutine,bi::definitely>;
template class bi::OverloadedDictionary<bi::MemberFunction,bi::definitely>;
template class bi::OverloadedDictionary<bi::BinaryOperator,bi::definitely>;
template class bi::OverloadedDictionary<bi::UnaryOperator,bi::definitely>;
template class bi::OverloadedDictionary<bi::AssignmentOperator,bi::definitely>;
