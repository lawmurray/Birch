/**
 * @file
 */
#include "libbirch/global.hpp"

#include "libbirch/World.hpp"

static std::random_device rd;

bi::World* bi::fiberWorld = new bi::World(0);
bool bi::fiberClone = false;
std::mt19937_64 bi::rng(rd());
