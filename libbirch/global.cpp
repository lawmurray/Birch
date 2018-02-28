/**
 * @file
 */
#include "libbirch/global.hpp"

#include "libbirch/World.hpp"

static std::random_device rd;

std::shared_ptr<bi::World> bi::fiberWorld = std::make_shared<bi::World>(0);
bool bi::fiberClone = false;
std::mt19937_64 bi::rng(rd());
