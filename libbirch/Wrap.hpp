/**
 * @file
 */
#pragma once

#include "libbirch/Enter.hpp"

namespace bi {
/**
 * Wraps the entry of a world around an object that is used.
 */
template<class T>
class Wrap {
public:
  /**
   * Constructor.
   *
   * @param o The object to wrap.
   * @param world The world in which enter.
   */
  Wrap(T& o, const std::shared_ptr<World>& world) :
      o(o),
      enter(world) {
    //
  }

  /**
   * Cast operator.
   */
  operator T&() {
    return o;
  }

private:
  T& o;
  Enter enter;
};

/**
 * Wrap an object with a world.
 */
template<class T>
Wrap<T> wrap(T& o, const std::shared_ptr<World>& world) {
  return Wrap<T>(o, world);
}
}
