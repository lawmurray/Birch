/**
 * @file
 */
#include "libbirch/global.hpp"

#include "libbirch/AllocationMap.hpp"
#include "libbirch/FiberWorld.hpp"

static std::random_device rd;

bi::FiberWorld* bi::fiberWorld = new FiberWorld(nullptr);
bi::AllocationMap* bi::allocationMap = new AllocationMap();

std::mt19937_64 bi::rng(rd());
