/**
 * @file
 */
#include "libbirch/global.hpp"

#include "libbirch/AllocationMap.hpp"

static std::random_device rd;

bi::AllocationMap* bi::fiberAllocationMap = new (GC) bi::AllocationMap();
size_t bi::fiberGen = 0;
std::mt19937_64 bi::rng(rd());
