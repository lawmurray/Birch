/**
 * @file
 */
#include "libbirch/World.hpp"

#include "libbirch/Any.hpp"

#include <cassert>

bi::World::World(const std::shared_ptr<World>& parent) :
    parent(parent) {
  //
}

std::shared_ptr<bi::Any> bi::World::get(const std::shared_ptr<Any>& src) {
  assert(src);
  if (this == src->getWorld()) {
    /* in this world */
    return src;
  } else if (!parent) {
    ///@todo Escaped pointer
    return src;
  } else {
    /* not in this world, propagate through ancestor world mappings */
    assert(parent);
    auto dst = parent->pull(src);

    /* apply own mapping, in case the object has been seen before; if it has
     * not then clone it and create a new mapping */
    auto iter = map.find(dst.get());
    if (iter != map.end()) {
      return iter->second;
    } else {
      assert(shared_from_this() == fiberWorld);
      auto clone = dst->clone();
      insert(dst, clone);
      return clone;
    }
  }
}

std::shared_ptr<bi::Any> bi::World::pull(
    const std::shared_ptr<Any>& src) const {
  assert(src);
  if (this == src->getWorld()) {
    /* in this world */
    return src;
  } else {
    /* not in this world, propagate through ancestor world mappings */
    assert(parent);
    auto dst = parent->pull(src);

    /* apply own mapping, in case the object has been seen before */
    auto iter = map.find(dst.get());
    if (iter != map.end()) {
      dst = iter->second;
    }
    return dst;
  }
}

void bi::World::insert(const std::shared_ptr<Any>& src,
    const std::shared_ptr<Any>& dst) {
  auto result = map.insert(std::make_pair(src.get(), dst));
  assert(result.second);
}
