/**
 * @file
 */
#include "libbirch/global.hpp"

#include "libbirch/AllocationMap.hpp"

static std::random_device rd;

bi::AllocationMap bi::allocationMap;
bi::world_t bi::fiberWorld = 0;
bi::world_t bi::nworlds = 0;
std::mt19937_64 bi::rng(rd());
