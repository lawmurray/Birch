/**
 * @file
 */
#include "bi/common/Dictionary.hpp"

#include "bi/expression/all.hpp"
#include "bi/statement/all.hpp"
#include "bi/type/all.hpp"
#include "bi/exception/all.hpp"

template<class ParameterType>
bool bi::Dictionary<ParameterType>::contains(ParameterType* param) const {
  auto iter = params.find(param->name->str());
  return iter != params.end() && iter->second == param;
}

template<class ParameterType>
bool bi::Dictionary<ParameterType>::contains(const std::string& name) const {
  return params.find(name) != params.end();
}

template<class ParameterType>
ParameterType* bi::Dictionary<ParameterType>::get(const std::string& name) {
  /* pre-condition */
  assert(contains(name));

  return params.find(name)->second;
}

template<class ParameterType>
void bi::Dictionary<ParameterType>::add(ParameterType* param) {
  /* pre-condition */
  assert(!contains(param));

  auto result = params.insert(std::make_pair(param->name->str(), param));
  assert(result.second);
}

template<class ParameterType>
void bi::Dictionary<ParameterType>::import(Dictionary<ParameterType>& o) {
  for (auto iter = o.params.begin(); iter != o.params.end(); ++iter) {
    if (!contains(iter->second)) {
      add(iter->second);
    }
  }
}

/*
 * Explicit instantiations.
 */
template class bi::Dictionary<bi::Parameter>;
template class bi::Dictionary<bi::GlobalVariable>;
template class bi::Dictionary<bi::LocalVariable>;
template class bi::Dictionary<bi::MemberVariable>;
template class bi::Dictionary<bi::Program>;
template class bi::Dictionary<bi::Basic>;
template class bi::Dictionary<bi::Class>;
template class bi::Dictionary<bi::Alias>;
