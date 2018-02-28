/**
 * @file
 */
#include "libbirch/Any.hpp"

#include "libbirch/global.hpp"

bi::Any::Any() : world(fiberWorld) {
  //
}

bi::Any::Any(const Any& o) : world(fiberWorld) {
  //
}

bi::Any::~Any() {
  //
}

std::shared_ptr<bi::Any> bi::Any::clone() const {
  return std::make_shared<Any>(*this);
}

const std::weak_ptr<bi::World>& bi::Any::getWorld() {
  return world;
}
