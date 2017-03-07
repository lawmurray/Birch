/**
 * @file
 */
#pragma once

#include <unordered_map>
#include <string>

namespace bi {
class Dispatcher;

/**
 * Dictionary for dispatchers.
 *
 * @ingroup compiler_common
 */
class DispatcherDictionary {
public:
  typedef std::unordered_map<std::string,Dispatcher*> map_type;

  /**
   * Does the dictionary contain the given dispatcher?
   */
  bool contains(Dispatcher* dispatcher);

  /**
   * If the dictionary contains an identical dispatcher, retrieve it.
   */
  Dispatcher* get(Dispatcher* dispatcher);

  /**
   * Add dispatcher.
   */
  void add(Dispatcher* dispatcher);

  /**
   * Merge another dictionary into this one.
   */
  void merge(DispatcherDictionary& o);

  /*
   * Iterator over dispatchers.
   */
  auto begin() {
    return dispatchers.begin();
  }
  auto end() {
    return dispatchers.end();
  }

  /**
   * Declarations within this scope.
   */
  map_type dispatchers;
};
}
