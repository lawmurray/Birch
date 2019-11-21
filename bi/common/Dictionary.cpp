/**
 * @file
 */
#include "bi/common/Dictionary.hpp"

#include "bi/common/Overloaded.hpp"
#include "bi/expression/all.hpp"
#include "bi/statement/all.hpp"
#include "bi/type/all.hpp"
#include "bi/exception/all.hpp"

template<class ObjectType>
bool bi::Dictionary<ObjectType>::contains(ObjectType* o) const {
  auto iter = objects.find(o->name->str());
  return iter != objects.end() && iter->second == o;
}

template<class ObjectType>
bool bi::Dictionary<ObjectType>::contains(const std::string& name) const {
  return objects.find(name) != objects.end();
}

template<class ObjectType>
ObjectType* bi::Dictionary<ObjectType>::get(const std::string& name) {
  /* pre-condition */
  assert(contains(name));

  return objects.find(name)->second;
}

template<class ObjectType>
void bi::Dictionary<ObjectType>::add(ObjectType* o) {
  /* pre-condition */
  assert(!contains(o->name->str()));

  auto result = objects.insert(std::make_pair(o->name->str(), o));
  assert(result.second);
}

template class bi::Dictionary<bi::Parameter>;
template class bi::Dictionary<bi::GlobalVariable>;
template class bi::Dictionary<bi::MemberVariable>;
template class bi::Dictionary<bi::LocalVariable>;
template class bi::Dictionary<bi::ForVariable>;
template class bi::Dictionary<bi::ParallelVariable>;
template class bi::Dictionary<bi::Program>;
template class bi::Dictionary<bi::Basic>;
template class bi::Dictionary<bi::Class>;
template class bi::Dictionary<bi::Generic>;
