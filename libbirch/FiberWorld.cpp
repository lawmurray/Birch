/**
 * @file
 */
#include "libbirch/FiberWorld.hpp"

bi::FiberWorld::FiberWorld(const FiberWorld* parent) :
    id(++nworlds),
    parent(parent) {
  //
}

size_t bi::FiberWorld::nworlds = 0;
