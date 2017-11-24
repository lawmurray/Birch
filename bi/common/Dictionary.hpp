/**
 * @file
 */
#pragma once

#include <unordered_map>
#include <string>

namespace bi {
/**
 * Dictionary for variables, functions, operators etc.
 *
 * @ingroup birch_common
 *
 * @tparam ObjectType Type of objects.
 */
template<class ObjectType>
class Dictionary {
public:
  typedef std::unordered_map<std::string,ObjectType*> map_type;

  /**
   * Does the dictionary contain the given object?
   */
  bool contains(ObjectType* param) const;

  /**
   * Does the dictionary contain the given object?
   */
  bool contains(const std::string& name) const;

  /**
   * Get a object by name.
   */
  ObjectType* get(const std::string& name);

  /**
   * Add object.
   *
   * @param o The object.
   */
  void add(ObjectType* o);

  /**
   * Resolve reference.
   *
   * @param[in,out] ref The reference.
   *
   * @return The object to which the reference can be resolved, or
   * `nullptr` if it cannot be resolved.
   */
  template<class ReferenceType>
  void resolve(ReferenceType* ref);

  /**
   * Iterators.
   */
  auto begin() const {
    return objects.begin();
  }
  auto end() const {
    return objects.end();
  }

  /**
   * Objects.
   */
  map_type objects;
};
}

template<class ObjectType>
template<class ReferenceType>
void bi::Dictionary<ObjectType>::resolve(ReferenceType* ref) {
  auto iter = objects.find(ref->name->str());
  if (iter != objects.end()) {
    ref->target = iter->second;
  }
}
