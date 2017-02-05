/**
 * @file
 */
#include "bi/common/OverloadedDictionary.hpp"

#include "bi/expression/FuncParameter.hpp"
#include "bi/expression/FuncReference.hpp"
#include "bi/exception/AmbiguousReferenceException.hpp"
#include "bi/exception/PreviousDeclarationException.hpp"

template<class ParameterType, class ReferenceType>
bool bi::OverloadedDictionary<ParameterType,ReferenceType>::contains(
    ParameterType* param) {
  auto iter = definites.find(param->name->str());
  return iter != definites.end() && iter->second.contains(param);
}

template<class ParameterType, class ReferenceType>
ParameterType* bi::OverloadedDictionary<ParameterType,ReferenceType>::get(
    ParameterType* param) {
  /* pre-condition */
  assert(contains(param));

  auto iter = definites.find(param->name->str());
  assert(iter != definites.end());
  return iter->second.get(param);
}

template<class ParameterType, class ReferenceType>
void bi::OverloadedDictionary<ParameterType,ReferenceType>::add(
    ParameterType* param) {
  /* pre-condition */
  assert(!contains(param));

  /* store in ordered list */
  this->ordered.push_back(param);

  /* store in definites poset */
  auto key1 = param->name->str();
  auto iter1 = definites.find(key1);
  if (iter1 != definites.end()) {
    auto& val1 = iter1->second;
    val1.insert(param);
  } else {
    auto val1 = definitely_poset_type();
    val1.insert(param);
    auto pair1 = std::make_pair(key1, val1);
    definites.insert(pair1);
  }

  /* store in possibles poset */
  auto key2 = param->name->str();
  auto iter2 = possibles.find(key2);
  if (iter2 != possibles.end()) {
    auto& val2 = iter2->second;
    val2.insert(param);
  } else {
    auto val2 = possibly_poset_type();
    val2.insert(param);
    auto pair2 = std::make_pair(key2, val2);
    possibles.insert(pair2);
  }
}

template<class ParameterType, class ReferenceType>
void bi::OverloadedDictionary<ParameterType,ReferenceType>::resolve(
    ReferenceType* ref) {
  /* definite matches */
  auto iter1 = definites.find(ref->name->str());
  if (iter1 == definites.end()) {
    ref->target = nullptr;
  } else {
    std::list<ParameterType*> definites;
    iter1->second.match(ref, definites);
    if (definites.size() > 1) {
      throw AmbiguousReferenceException(ref, definites);
    } else if (definites.size() == 1) {
      ref->target = definites.front();
    } else {
      ref->target = nullptr;
    }
  }

  /* possible matches */
  auto iter2 = possibles.find(ref->name->str());
  if (iter2 == possibles.end()) {
    ref->alternatives.clear();
  } else {
    std::list<ParameterType*> possibles;
    iter2->second.match(ref, possibles);
    ref->alternatives = possibles;
  }
}

/*
 * Explicit instantiations.
 */
template class bi::OverloadedDictionary<bi::FuncParameter,bi::FuncReference>;
