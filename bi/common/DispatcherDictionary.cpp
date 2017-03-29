/**
 * @file
 */
#include "bi/common/DispatcherDictionary.hpp"

#include "bi/dispatcher/Dispatcher.hpp"

bool bi::DispatcherDictionary::contains(Dispatcher* dispatcher) {
  auto range = dispatchers.equal_range(dispatcher->name->str());
  for (auto iter = range.first; iter != range.second; ++iter) {
    if (*iter->second == *dispatcher) {
      return true;
    }
  }
  return false;
}

bi::Dispatcher* bi::DispatcherDictionary::get(Dispatcher* dispatcher) {
  auto range = dispatchers.equal_range(dispatcher->name->str());
  for (auto iter = range.first; iter != range.second; ++iter) {
    if (*iter->second == *dispatcher) {
      return iter->second;
    }
  }
  return nullptr;
}

void bi::DispatcherDictionary::add(Dispatcher* dispatcher) {
  /* pre-condition */
  assert(!contains(dispatcher));

  dispatchers.insert(std::make_pair(dispatcher->name->str(), dispatcher));
}

void bi::DispatcherDictionary::merge(DispatcherDictionary& o) {
  for (auto iter = o.dispatchers.begin(); iter != o.dispatchers.end();
      ++iter) {
    if (!contains(iter->second)) {
      add(iter->second);
    }
  }
}
