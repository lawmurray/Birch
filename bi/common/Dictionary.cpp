/**
 * @file
 */
#include "bi/common/Dictionary.hpp"

#include "bi/expression/all.hpp"
#include "bi/statement/all.hpp"
#include "bi/type/all.hpp"
#include "bi/expression/all.hpp"
#include "bi/exception/all.hpp"

template<class ParameterType>
bool bi::Dictionary<ParameterType>::contains(ParameterType* param) {
  return params.find(param->name->str()) != params.end();
}

template<class ParameterType>
ParameterType* bi::Dictionary<ParameterType>::get(ParameterType* param) {
  /* pre-condition */
  assert(contains(param));

  return params.find(param->name->str())->second;
}

template<class ParameterType>
void bi::Dictionary<ParameterType>::add(ParameterType* param) {
  /* pre-condition */
  assert(!contains(param));

  auto result = params.insert(std::make_pair(param->name->str(), param));
  assert(result.second);
}

template<class ParameterType>
void bi::Dictionary<ParameterType>::merge(Dictionary<ParameterType>& o) {
  for (auto iter = o.params.begin(); iter != o.params.end(); ++iter) {
    if (!contains(iter->second)) {
      add(iter->second);
    }
  }
}

/*
 * Explicit instantiations.
 */
template class bi::Dictionary<bi::VarParameter>;
template class bi::Dictionary<bi::TypeParameter>;
template class bi::Dictionary<bi::ProgParameter>;
